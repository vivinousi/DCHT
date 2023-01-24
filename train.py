import argparse
import os
import torch
from torch.utils.data import DataLoader
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR

from tracker.dchnet import DCHT
from utils.loss import BCEWeightedLoss
from utils.train_data import CocoTracking, create_labels_and_weights


def main(args):
    train_dataset = CocoTracking(root=os.path.join(args.data_root, 'train2017'),
                                 annFile=os.path.join(args.data_root, 'annotations', 'instances_train2017.json'))
    val_dataset = CocoTracking(root=os.path.join(args.data_root, 'val2017'),
                               annFile=os.path.join(args.data_root, 'annotations', 'instances_val2017.json'))

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True,
                                  drop_last=True, num_workers=args.num_workers)
    train_iters = len(train_dataloader)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True,
                                drop_last=True, num_workers=args.num_workers)

    dcht_net = DCHT()
    dcht_net = dcht_net.to(args.device)
    dcht_net.initialize('weights/pretrained_backbone.pth')

    # template_size = 127
    # search_size = 255
    # z = torch.ones(1, 3, int(template_size), int(template_size)).to(args.device)
    # x = torch.ones(1, 3, int(search_size), int(search_size)).to(args.device)
    output_size, total_stride = dcht_net.output_size, dcht_net.total_stride

    optimizer = SGD(dcht_net.get_params(args.lr, args.weight_decay),
                    lr=args.lr, weight_decay=args.weight_decay)
    optimizer_ft = SGD(dcht_net.get_finetuning_params(args.lr * 10, args.weight_decay),
                       lr=args.lr * 10, weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, 1, gamma=0.99)
    criterion = BCEWeightedLoss()

    t, w = create_labels_and_weights(output_size, r_pos=16, r_neg=0, total_stride=total_stride)
    t = t.repeat(args.batch_size, 1, 1, 1).to(args.device)
    w = w.repeat(args.batch_size, 1, 1, 1).to(args.device)

    n_epochs = args.epochs

    for epoch in range(n_epochs // 4):
        loss_epoch = 0
        for idx, batch in enumerate(train_dataloader):
            z, x = batch[0].to(args.device), batch[1].to(args.device)

            optimizer_ft.zero_grad()
            pred = dcht_net(z, x)
            loss = criterion(pred, t, w)
            loss.backward()
            optimizer_ft.step()

            loss_epoch += loss.item()

            if idx % args.log_freq == 0:
                print(
                    'Finetuning Epoch: {:3d}/{:3d} || Iter: {:4d}/{:4d} || Iter Loss: {:.6f} '
                    '|| Avg Loss: {:.6f}'.format(
                        epoch + 1, n_epochs // 4, idx + 1, train_iters, loss, loss_epoch / (idx + 1)))

    for epoch in range(n_epochs - n_epochs // 4):
        loss_epoch = 0
        for idx, batch in enumerate(train_dataloader):
            z, x = batch[0].to(args.device), batch[1].to(args.device)

            optimizer.zero_grad()
            pred, feat_x, feat_z = dcht_net(z, x)
            loss = criterion(pred, t, w)
            loss.backward()
            optimizer.step()

            loss_epoch += loss.item()

            if idx % args.log_freq == 0:
                print(
                    'Training Epoch: {:3d}/{:3d} || Iter: {:4d}/{:4d} || Iter Loss: {:.6f} '
                    '|| Avg Loss: {:.6f} || LR: {:.6f}'.format(
                        epoch + 1, n_epochs, idx + 1, train_iters, loss, loss_epoch / (idx + 1),
                        scheduler.get_last_lr()[0]))
        scheduler.step()
        # validation
        val_loss = 0
        with torch.no_grad():
            for idx, batch in enumerate(val_dataloader):
                z, x = batch[0].to(args.device), batch[1].to(args.device)
                pred, feat_x, feat_z = dcht_net(z, x)
                loss = criterion(pred, t, w)
                val_loss += loss.item()
        print('Validation Avg Loss: {:.6f}'.format(val_loss / len(val_dataloader)))

        torch.save(dcht_net.state_dict(), 'weights/trained_model.pth')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs.')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size for training.')
    parser.add_argument('--num-workers', type=int, default=8, help='Number of threads used for data loading.')
    parser.add_argument('--log-freq', type=int, default=20,  help='Logging frequency (iterations)')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device used for training')
    parser.add_argument('--lr', type=float, default=1e-2, help='Initial learning rate.')
    parser.add_argument('--weight-decay', type=float, default=5e-4, help='Weight decay rate.')
    parser.add_argument('--data-root', type=str, default='data/coco', help='Dataset root path.')

    args = parser.parse_args()
    main(args)
