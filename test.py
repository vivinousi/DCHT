import argparse
import os
import time
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import scipy.io as sio

from tracker.dchnet import DCHT
from utils.test_data import UAVDatasetReader, VOTDatasetReader, OTBDatasetReader
from utils.track_utils import rect1_2_cxy_wh, center_error, iou


def main(args):
    if args.data_type == 'uav123' or args.data_type == 'uav20l':
        dataset_reader = UAVDatasetReader(args.data_root, subset=args.data_type)
    elif args.data_type == 'otb':
        dataset_reader = OTBDatasetReader(args.data_root)
    else:
        dataset_reader = VOTDatasetReader(args.data_root)
    tracker = DCHT().eval().to(args.device)
    tracker.load_state_dict(torch.load(args.weights))

    test_path = os.path.join('results', args.data_type.upper())
    if args.write_mat:
        os.makedirs(test_path, exist_ok=True)

    avg_iou = 0.
    avg_prec = 0.
    avg_speed = 0.

    prec_thresholds = np.linspace(0, 50, 100)
    iou_thresholds = np.linspace(0, 1, 100)
    avg_success = np.zeros((100,))
    avg_precision = np.zeros((100,))

    for video_idx, (video_name, video_reader) in enumerate(dataset_reader):
        pred_bbs = []
        n_frames = len(video_reader)
        tic = time.time()  # time start
        for frame_idx, (frame, gt_bb) in enumerate(video_reader):
            if frame_idx == 0:
                pred_bb = tracker.init(frame, rect1_2_cxy_wh(gt_bb))
            else:
                pred_bb = tracker.update(frame)
            if args.show:
                im_show = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                cv2.rectangle(im_show, (int(pred_bb[0]), int(pred_bb[1])),
                              (int(pred_bb[0] + pred_bb[2]), int(pred_bb[1] + pred_bb[3])),
                              (60, 20, 220), 4)
                cv2.putText(im_show, str(frame_idx), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2,
                            cv2.LINE_AA)
                cv2.imshow(video_name, im_show)
                k = cv2.waitKey(1)
                if k == ord('q'):
                    break
                elif k == ord('w'):
                    exit()
            pred_bbs.append(pred_bb)
        toc = time.time() - tic

        cv2.destroyAllWindows()

        # video metrics
        speed_fps = n_frames / toc
        pred_bbs = np.asarray(pred_bbs)
        pred_bbs[:, :2] += 1
        gt_bbs = video_reader.get_groundtruth()
        pred_bbs = pred_bbs[~np.isnan(gt_bbs).any(axis=1)]
        gt_bbs = gt_bbs[~np.isnan(gt_bbs).any(axis=1)]

        n_boxes = float(len(gt_bbs))
        ious = iou(pred_bbs, gt_bbs)
        mean_iou = ious.mean()
        center_errors = center_error(pred_bbs, gt_bbs)
        prec = (center_errors <= 20).sum() / n_boxes

        success = np.asarray([np.sum(ious > t) / n_boxes for t in iou_thresholds])
        precision = np.asarray([np.sum(center_errors <= t) / n_boxes for t in prec_thresholds])

        # dataset metrics
        avg_iou += mean_iou
        avg_success += success
        avg_precision += precision
        avg_prec += prec
        avg_speed += speed_fps

        print('{:3d}/{:3d} Performance on {:20s}:\tIoU: {:1.3f} '
              '\tPrec: {:1.3f}\tSpeed: {:3.3f} fps'.format(
            video_idx + 1, len(dataset_reader), video_name, mean_iou, prec, speed_fps))

        if args.write_mat:
            out_filename = os.path.join(test_path, f'{video_name}.mat')
            mat_data = {'res': pred_bbs,
                        'len': pred_bbs.shape[0],
                        'type': 'rect',
                        'fps': speed_fps}
            sio.savemat(out_filename, {'results': [mat_data]})

    avg_iou /= len(dataset_reader)
    avg_prec /= len(dataset_reader)
    avg_speed /= len(dataset_reader)

    avg_success /= len(dataset_reader)
    auc = np.trapz(avg_success, iou_thresholds)
    avg_precision /= len(dataset_reader)

    plt.figure()
    plt.plot(iou_thresholds, avg_success, linewidth=4, label=f'DCHT [{auc:1.3f}]')
    plt.title(f'Success plot - {args.data_type.upper()}')
    plt.legend()
    plt.savefig('success_plot.png')

    plt.figure()
    plt.plot(prec_thresholds, avg_precision, linewidth=4, label=f'DCHT [{avg_prec:1.3f}]')
    plt.title(f'Precision plot - {args.data_type.upper()}')
    plt.legend()
    plt.savefig('precision_plot.png')

    plt.show()

    print('Overall performance:  IoU: {:1.3f} Precision {:1.3f} Speed: {:3.3f} fps'.format(
        avg_iou, avg_prec, avg_speed))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0', help='Device used for testing.')
    parser.add_argument('--weights', type=str, default='weights/pretrained_model.pth', help='Path to model weights.')
    parser.add_argument('--data-type', type=str, default='uav123', help='Dataset type.',
                        choices=['uav123', 'uav20l', 'vot', 'otb'])
    parser.add_argument('--data-root', type=str, default='data/uav123', help='Dataset root path.')
    parser.add_argument('--show', action='store_true', help='Visualize result.')
    parser.add_argument('--write-mat', action='store_true', help='Write result to mat file to be read by benchmark '
                                                                 'toolkits.')

    args = parser.parse_args()
    main(args)
