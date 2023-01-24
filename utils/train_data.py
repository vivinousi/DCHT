import os
from collections import namedtuple

import cv2
import imgaug
import xmltodict
import numpy as np
from tqdm import tqdm

np.random.seed(43)
import random
from torch.utils.data import Dataset
import torch
from torchvision.datasets import coco
from torchvision.transforms import Compose, Lambda, Normalize

from utils.track_utils import get_subwindow_tracking, rect0_2_cxy_wh


def get_video_names(data_path):
    if os.path.exists(data_path):
        return [dI for dI in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, dI))]
    else:
        return []


normalize = Compose([Lambda(lambda x: x / 255.0),
                     Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])])

AugmentationOpts = namedtuple("AugmentationOpts", ["hsv", "motion_blur", "flip", "rotate",
                                                   "brightness", "range", "context_amount",
                                                   "search_sz", "template_sz", "grayscale",
                                                   "normalize"])
default_opts = AugmentationOpts(hsv=True, motion_blur=0.01, flip=0, rotate=0, brightness=0.25, range=10,
                                context_amount=0.5, search_sz=255, template_sz=127, grayscale=0.01,
                                normalize=False)


class Got10k(Dataset):
    def __init__(self, root='data/got10k/', opts=default_opts):
        self.root = os.path.join(root, 'train')
        self.videos = get_video_names(self.root)
        self.videos.remove('GOT-10k_Train_004419')
        self.opts = opts
        self.images = {video: [] for video in self.videos}
        for video in tqdm(self.videos):
            for dirpath, _, filenames in os.walk(os.path.join(self.root, video)):
                for filename in filenames:
                    if filename.endswith('.jpg'):
                        self.images[video].append(os.path.join(dirpath, filename))

    def __getitem__(self, item):
        video = self.videos[item]
        boxes = np.loadtxt(os.path.join(self.root, video, 'groundtruth.txt'), delimiter=',')

        img_z, img_x = self.transform(self.images[video], boxes)
        return img_z, img_x

    def __len__(self):
        return len(self.videos)

    def transform(self, images, boxes):
        box_z = np.asarray([-1, -1, -1, -1])
        while box_z[3] == -1 or box_z[2] == -1:
            target_idx = np.random.randint(len(images))
            anno_target_idx = int(images[target_idx].split('/')[-1].split('.')[0]) - 1
            box_z = boxes[anno_target_idx]

        box_x = np.asarray([-1, -1, -1, -1])
        while box_x[3] == -1 or box_x[2] == -1:
            search_idx = max(min(np.random.randint(target_idx - self.opts.range,
                                                   target_idx + self.opts.range), len(images) - 1), 0)
            anno_search_idx = int(images[search_idx].split('/')[-1].split('.')[0]) - 1
            box_x = boxes[anno_search_idx]

        img_z = cv2.cvtColor(cv2.imread(images[target_idx]), cv2.COLOR_BGR2RGB)
        img_x = cv2.cvtColor(cv2.imread(images[search_idx]), cv2.COLOR_BGR2RGB)

        img_z, img_x = get_crops(img_z, box_z, img_x, box_x, self.opts)
        return img_z, img_x


class TrackingNetTrain(Dataset):
    def __init__(self, root='data/TrackingNet-devkit/data/',
                 opts=default_opts, n_folders=12):
        self.root = root
        self.train_folders = ['TRAIN_{}'.format(i) for i in range(n_folders)]

        self.videos = []
        for i in range(n_folders):
            for video in get_video_names(os.path.join(self.root, self.train_folders[i], 'frames')):
                self.videos.append((i, video))
        self.images = {video[1]: [] for video in self.videos}
        for f, video in tqdm(self.videos):
            for dirpath, _, filenames in os.walk(os.path.join(self.root, self.train_folders[f], 'frames', video)):
                for filename in filenames:
                    self.images[video].append(os.path.join(dirpath, filename))
        self.opts = opts

    def __getitem__(self, item):
        f, video = self.videos[item]

        boxes = np.loadtxt(os.path.join(self.root, self.train_folders[f], 'anno', '{}.txt'.format(video)),
                           delimiter=',')

        img_z, img_x = self.transform(self.images[video], boxes)
        return img_z, img_x

    def transform(self, images, boxes):
        target_idx = np.random.randint(len(images))
        search_idx = max(min(np.random.randint(target_idx - self.opts.range,
                                               target_idx + self.opts.range), len(images) - 1), 0)

        img_z = cv2.cvtColor(cv2.imread(images[target_idx]), cv2.COLOR_BGR2RGB)
        img_x = cv2.cvtColor(cv2.imread(images[search_idx]), cv2.COLOR_BGR2RGB)
        anno_target_idx = int(images[target_idx].split('/')[-1].split('.')[0])
        anno_search_idx = int(images[search_idx].split('/')[-1].split('.')[0])
        box_z = boxes[anno_target_idx]
        box_x = boxes[anno_search_idx]

        img_z, img_x = get_crops(img_z, box_z, img_x, box_x, self.opts)
        return img_z, img_x

    def __len__(self):
        return len(self.videos)


class VisDrone2018SOT(Dataset):
    def __init__(self, base_path='data/VisDrone2018-SOT-toolkit/data/', subset='train',
                 opts=None):
        assert subset in ['train', 'val'], "Unrecognized subset {},must be one of" \
                                                             "['train', 'val']".format(subset)
        self.folder_path = os.path.join(base_path, 'VisDrone2018-SOT-{}'.format(subset), 'sequences')
        self.video_names = get_video_names(self.folder_path)
        self.annotations_path = os.path.join(base_path, 'VisDrone2018-SOT-{}'.format(subset), 'annotations')
        self.len = len(self.video_names)
        self.opts = opts

    def __getitem__(self, item):
        images = sorted([os.path.join(self.folder_path, self.video_names[item], filename)
                         for filename in os.listdir(os.path.join(self.folder_path, self.video_names[item]))
                         if filename.endswith('.jpg')])
        gt = np.genfromtxt(os.path.join(self.annotations_path, '{}.txt'.format(self.video_names[item])), delimiter=',',
                           dtype=np.int32)
        img_z, img_x = self.transform(images, gt)
        return img_z, img_x

    def transform(self, images, gt):
        target_idx = np.random.randint(len(images))
        search_idx = max(min(np.random.randint(target_idx - self.opts.range,
                                               target_idx + self.opts.range), len(images) - 1), 0)
        img_z = cv2.cvtColor(cv2.imread(images[target_idx]), cv2.COLOR_BGR2RGB)
        img_x = cv2.cvtColor(cv2.imread(images[search_idx]), cv2.COLOR_BGR2RGB)
        avg_color = [int(x) for x in np.mean(img_z, axis=(0, 1))]
        center_z, size_z = rect0_2_cxy_wh(gt[target_idx])
        center_x, size_x = rect0_2_cxy_wh(gt[search_idx])

        mean_dim_z = self.opts.context_amount * (np.sum(size_z))
        z_sz = np.sqrt((size_z[0] + mean_dim_z) * (size_z[1] + mean_dim_z))
        mean_dim_x = self.opts.context_amount * (np.sum(size_x))
        x_sz = np.sqrt((size_x[0] + mean_dim_x) * (size_x[1] + mean_dim_x))
        x_sz = float(self.opts.search_sz) / self.opts.template_sz * x_sz

        img_x = get_subwindow_tracking(img_x, center_x, self.opts.search_sz, x_sz, avg_color)
        img_z = get_subwindow_tracking(img_z, center_z, self.opts.template_sz, z_sz, avg_color)

        assert img_z.shape[0] == self.opts.template_sz
        assert img_x.shape[0] == self.opts.search_sz

        img_z, img_x = augment(img_z, img_x, self.opts)

        img_z = np.transpose(img_z, (2, 0, 1)).astype(np.float32)
        img_x = np.transpose(img_x, (2, 0, 1)).astype(np.float32)
        return img_z, img_x

    def __len__(self):
        return self.len


class ImageNetTracking(Dataset):
    def __init__(self, root, subset='train', opts=None):
        self.root = root
        self.subset = subset
        self.images_path = os.path.join(self.root, 'Data/DET/{}'.format(subset))
        self.annotations_path = os.path.join(self.root, 'Annotations/DET/{}'.format(subset))
        self.filenames = [os.path.join(os.path.relpath(root, self.images_path), name)
                          for root, dirs, files in os.walk(self.images_path) for name in files
                          if name.endswith((".JPEG", ".jpg", ".JPG"))]
        random.shuffle(self.filenames)
        self.opts = opts

        self.filter_filenames()

    def filter_filenames(self):
        for filename in self.filenames:
            anno_path = os.path.join(self.annotations_path, filename.split('.')[0] + '.xml')
            with open(anno_path, 'r') as f:
                xml_string = f.read()
            annotations = xmltodict.parse(xml_string)
            if 'object' not in annotations['annotation']:
                self.filenames.remove(filename)

    def __getitem__(self, item):
        image_path = os.path.join(self.images_path, self.filenames[item])
        anno_path = os.path.join(self.annotations_path, self.filenames[item].split('.')[0] + '.xml')
        with open(anno_path, 'r') as f:
            xml_string = f.read()
        annotations = xmltodict.parse(xml_string)
        objects = annotations['annotation']['object']
        if not type(objects) == list:
            objects = [objects]
        boxes = [(int(obj['bndbox']['xmin']), int(obj['bndbox']['ymin']),
                  int(obj['bndbox']['xmax']) - int(obj['bndbox']['xmin']),
                  int(obj['bndbox']['ymax']) - int(obj['bndbox']['ymin'])) for obj in objects]
        classes = [obj['name'] for obj in objects]

        img_z, img_x = self.transform(image_path, boxes, classes)
        return img_z, img_x

    def transform(self, im_path, boxes):
        img = cv2.cvtColor(cv2.imread(im_path), cv2.COLOR_BGR2RGB)
        target_idx = np.random.randint(0, len(boxes))
        target_box = boxes[target_idx]
        search_idx = target_idx
        search_box = boxes[search_idx]

        img_z, img_x = get_crops(img, np.asarray(target_box), img, np.asarray(search_box), self.opts)
        return img_z, img_x

    def __len__(self):
        return len(self.filenames)


class CocoTracking(Dataset):
    def __init__(self, root, annFile, opts=default_opts):
        super(CocoTracking, self).__init__()
        self.coco = coco.CocoDetection(root, annFile)
        self.opts = opts

    def __len__(self):
        return len(self.coco)

    def __getitem__(self, item):
        data = self.coco[int(item)]
        i = 1
        while len(data[1]) == 0:
            data = self.coco[int(item) + i]
            i += 1
        img_z, img_x = self.transform(data)

        return img_z, img_x

    def transform(self, data):
        target_idx = np.random.randint(0, len(data[1]))

        search_data = data
        search_idx = target_idx

        img_z = np.array(data[0])
        img_x = np.array(search_data[0])

        avg_color = [int(x) for x in np.mean(img_z, axis=(0, 1))]
        center_z, size_z = rect0_2_cxy_wh(data[1][target_idx]['bbox'])
        center_x, size_x = rect0_2_cxy_wh(search_data[1][search_idx]['bbox'])

        mean_dim_z = self.opts.context_amount * (np.sum(size_z))
        z_sz = np.sqrt((size_z[0] + mean_dim_z) * (size_z[1] + mean_dim_z))

        mean_dim_x = self.opts.context_amount * (np.sum(size_x))
        x_sz = np.sqrt((size_x[0] + mean_dim_x) * (size_x[1] + mean_dim_x))
        x_sz = float(self.opts.search_sz) / self.opts.template_sz * x_sz

        img_z = get_subwindow_tracking(img_z, center_z, self.opts.template_sz, z_sz, avg_color)
        img_x = get_subwindow_tracking(img_x, center_x, self.opts.search_sz, x_sz, avg_color)

        img_z, img_x = augment(img_z, img_x, self.opts)

        img_z = np.transpose(img_z, (2, 0, 1)).astype(np.float32)
        img_x = np.transpose(img_x, (2, 0, 1)).astype(np.float32)

        if self.opts.normalize:
            img_z = normalize(torch.from_numpy(img_z))
            img_x = normalize(torch.from_numpy(img_x))

        return img_z, img_x


def get_crop(im, bbox, size_z, size_x, context_amount):
    cy, cx, h, w = bbox[1] + bbox[3] * 0.5, bbox[0] + bbox[2] * 0.5, bbox[3], bbox[2]
    wc_z = w + context_amount * (w + h)
    hc_z = h + context_amount * (w + h)
    s_z = np.sqrt(wc_z * hc_z)
    scale_z = size_z / s_z

    d_search = (size_x - size_z) / 2
    pad = d_search / scale_z
    s_x = s_z + 2 * pad
    scale_x = size_x / s_x

    image_crop_x, _, _, _, _ = get_subwindow_avg(im, [cy, cx],
                                                 [size_x, size_x],
                                                 [np.round(s_x), np.round(s_x)])

    return image_crop_x, scale_x


def get_center(x):
    return (x - 1.) / 2.


def get_subwindow_avg(im, pos, model_sz, original_sz):
    # avg_chans = np.mean(im, axis=(0, 1)) # This version is 3x slower
    avg_chans = [np.mean(im[:, :, 0]), np.mean(im[:, :, 1]), np.mean(im[:, :, 2])]
    if not original_sz:
        original_sz = model_sz
    sz = original_sz
    im_sz = im.shape
    # make sure the size is not too small
    assert im_sz[0] > 2 and im_sz[1] > 2
    c = [get_center(s) for s in sz]

    # check out-of-bounds coordinates, and set them to avg_chans
    context_xmin = np.int(np.round(pos[1] - c[1]))
    context_xmax = np.int(context_xmin + sz[1] - 1)
    context_ymin = np.int(np.round(pos[0] - c[0]))
    context_ymax = np.int(context_ymin + sz[0] - 1)
    left_pad = np.int(np.maximum(0, -context_xmin))
    top_pad = np.int(np.maximum(0, -context_ymin))
    right_pad = np.int(np.maximum(0, context_xmax - im_sz[1] + 1))
    bottom_pad = np.int(np.maximum(0, context_ymax - im_sz[0] + 1))

    context_xmin = context_xmin + left_pad
    context_xmax = context_xmax + left_pad
    context_ymin = context_ymin + top_pad
    context_ymax = context_ymax + top_pad
    if top_pad > 0 or bottom_pad > 0 or left_pad > 0 or right_pad > 0:
        R = np.pad(im[:, :, 0], ((top_pad, bottom_pad), (left_pad, right_pad)),
                   'constant', constant_values=(avg_chans[0]))
        G = np.pad(im[:, :, 1], ((top_pad, bottom_pad), (left_pad, right_pad)),
                   'constant', constant_values=(avg_chans[1]))
        B = np.pad(im[:, :, 2], ((top_pad, bottom_pad), (left_pad, right_pad)),
                   'constant', constant_values=(avg_chans[2]))

        im = np.stack((R, G, B), axis=2)

    im_patch_original = im[context_ymin:context_ymax + 1,
                        context_xmin:context_xmax + 1, :]
    if not (model_sz[0] == original_sz[0] and model_sz[1] == original_sz[1]):
        im_patch = cv2.resize(im_patch_original, tuple(model_sz), cv2.INTER_LINEAR)
    else:
        im_patch = im_patch_original
    return im_patch, left_pad, top_pad, right_pad, bottom_pad


def get_crops(img_z, box_z, img_x, box_x, opts):
    avg_color = [int(x) for x in np.mean(img_z, axis=(0, 1))]
    center_z, size_z = rect0_2_cxy_wh(box_z)
    center_x, size_x = rect0_2_cxy_wh(box_x)

    context_z = opts.context_amount * (np.sum(size_z))
    z_sz = np.sqrt((size_z[0] + context_z) * (size_z[1] + context_z))

    r = np.sqrt(np.prod(size_z)/np.prod(size_x))
    try:
        img_x = cv2.resize(img_x, dsize=None, fx=r, fy=r)
    except:
        print('unable to resize')
    size_x *= r
    center_x *= r
    context_x = opts.context_amount * np.sum(size_x)
    x_sz = np.sqrt((size_x[0] + context_x) * (size_x[1] + context_x))
    x_sz = (float(opts.search_sz) / opts.template_sz) * x_sz

    img_z = get_subwindow_tracking(img_z, center_z, opts.template_sz, z_sz, avg_color)
    img_x = get_subwindow_tracking(img_x, center_x, opts.search_sz, x_sz, avg_color)

    img_z, img_x = augment(img_z, img_x, opts)

    img_z = np.transpose(img_z, (2, 0, 1)).astype(np.float32)
    img_x = np.transpose(img_x, (2, 0, 1)).astype(np.float32)

    if opts.normalize:
        img_z = normalize(torch.from_numpy(img_z))
        img_x = normalize(torch.from_numpy(img_x))
    return img_z, img_x


def augment(img_z, img_x, opts):
    if np.random.rand() < opts.flip:
        if np.random.rand() < 0.5:
            img_z = cv2.flip(img_z, 1)
        else:
            img_x = cv2.flip(img_x, 1)

    if np.random.rand() < opts.grayscale:
        grayer = imgaug.augmenters.Grayscale(alpha=(0.5, 1.0))
        img_z = grayer.augment_image(img_z)
        img_x = grayer.augment_image(img_x)

    if np.random.rand() < opts.rotate:
        rotator = imgaug.augmenters.Affine(rotate=(-20, 20))
        img_z = rotator.augment_image(img_z)
        img_x = rotator.augment_image(img_x)

    if opts.hsv:
        adder = imgaug.augmenters.AddToHueAndSaturation((-10, 10))
        img_z = adder.augment_image(img_z)
        img_x = adder.augment_image(img_x)

    if np.random.rand() < opts.motion_blur:
        size = 7
        # generating the kernel
        kernel_motion_blur = np.zeros((size, size))
        kernel_motion_blur[int((size - 1) / 2), :] = np.ones(size)
        kernel_motion_blur = kernel_motion_blur / size
        # applying the kernel to the input image
        if np.random.rand() < 0.5:
            img_z = cv2.filter2D(img_z, -1, kernel_motion_blur)
        else:
            img_x = cv2.filter2D(img_x, -1, kernel_motion_blur)

    if np.random.rand() < opts.brightness:
        aug = imgaug.augmenters.Multiply(mul=(0.75, 1.25))
        img_x = aug.augment_image(img_x)

    return img_z, img_x


def create_logisticloss_labels(output_sz, r_pos, r_neg, total_stride, ignore_label=-100):
    label_sz = output_sz
    r_pos = r_pos / total_stride
    r_neg = r_neg / total_stride
    labels = np.zeros((label_sz, label_sz))

    for r in range(label_sz):
        for c in range(label_sz):
            dist = np.sqrt((r - label_sz // 2) ** 2 +
                           (c - label_sz // 2) ** 2)
            if dist <= r_pos:
                labels[r, c] = 1
            elif dist <= r_neg:
                labels[r, c] = ignore_label
            else:
                labels[r, c] = 0
    return labels


def create_labels_and_weights(output_sz, r_pos, r_neg, total_stride):
    labels = create_logisticloss_labels(output_sz, r_pos, r_neg, total_stride)
    weights = np.zeros_like(labels)

    neg_label = 0
    pos_num = np.sum(labels == 1)
    neg_num = np.sum(labels == neg_label)
    if pos_num > 0:
        weights[labels == 1] = 0.5 / pos_num
    weights[labels == neg_label] = 0.5 / neg_num
    weights *= pos_num + neg_num

    return torch.from_numpy(labels.astype(np.float32)).unsqueeze_(0).unsqueeze_(0), \
           torch.from_numpy(weights.astype(np.float32)).unsqueeze_(0).unsqueeze_(0)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    opts = AugmentationOpts(hsv=True, motion_blur=0.05, flip=0, rotate=0, brightness=0.25, range=25)

    # data = TrackingNet(n_folders=1)
    # data = Got10k(opts=opts)
    # data = ImageNetTracking(root='/media/FastData/datasets/ILSVRC')
    # data = VisDrone2018SOT(base_path='/media/data/vivi/tracking/toolkits/VisDrone2018-SOT-toolkit/data/', opts=opts)
    data = CocoTracking(root='/home/administrator/data/coco/train2017/',
                        annFile='/home/administrator/data/coco/annotations/instances_train2017.json',
                        opts=opts)
    n = len(data)
    fig = plt.figure(1)
    ax = fig.add_axes([0, 0, 1, 1])

    for i in range(n):
        z, x, t, w = data[i]
        z = z.transpose(1, 2, 0).astype(np.uint8)
        x = x.transpose(1, 2, 0).astype(np.uint8)
        print(z.shape, x.shape, t.shape, t.min(), t.max())
        cv2.circle(x, (127, 127), 64, (0, 255, 0), 2)
        cv2.imshow('z', cv2.cvtColor(np.uint8(z), cv2.COLOR_BGR2RGB))
        cv2.imshow('x', cv2.cvtColor(np.uint8(x), cv2.COLOR_BGR2RGB))
        cv2.imshow('t', np.uint8(t.transpose(1, 2, 0) * 255))
        cv2.waitKey(0)
    cv2.destroyAllWindows()
