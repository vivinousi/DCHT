import os
import json
import cv2
import numpy as np
from torch.utils.data import Dataset

from utils.track_utils import _poly


def get_video_names(data_path):
    if os.path.exists(data_path):
        return sorted(
            [dI for dI in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, dI)) and 'test' not in dI])
    else:
        return []


class OTBVideoReader(Dataset):
    def __init__(self, root, video_info, preload=False):
        self.root = root
        self.video_info = video_info
        self.preload = preload
        if self.preload:
            n_frames = len(self.video_info["img_names"])
            img = cv2.imread(os.path.join(self.root, self.video_info["img_names"][0]))
            h, w, c = img.shape
            images = np.zeros((n_frames, h, w, c))
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            elif img.ndim == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images[0, :, :, :] = img
            for item in range(n_frames):
                img = cv2.imread(os.path.join(self.root, self.video_info["img_names"][item]))
                if img.ndim == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                elif img.ndim == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                images[item, :, :, :] = img
            self.images = images

    def __getitem__(self, item):
        if self.preload:
            img = self.images[item]
        else:
            img = cv2.imread(os.path.join(self.root, self.video_info["img_names"][item]))
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            elif img.ndim == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        gt_bbox = self.video_info["gt_rect"][item]
        return img, gt_bbox

    def __len__(self):
        return len(self.video_info["img_names"])

    def get_groundtruth(self):
        return np.asarray(self.video_info["gt_rect"])


class OTBDatasetReader(Dataset):
    def __init__(self, root, json_filename='OTB2015.json', preload=False):
        self.root = root
        with open(os.path.join(root, json_filename)) as f:
            self.dataset_info = json.load(f)
        self.videos = list(self.dataset_info.keys())
        # self.videos = self.videos[:4]
        self.preload = preload

    def __getitem__(self, item):
        return self.videos[item], OTBVideoReader(self.root, self.dataset_info[self.videos[item]],
                                                 preload=self.preload)

    def __len__(self):
        return len(self.videos)


class UAVVideoReader(Dataset):
    def __init__(self, root, subset, sequence, preload=False):
        self.anno_path = os.path.join(root, 'anno', subset, '{}.txt'.format(sequence))
        self.groundtruth = np.genfromtxt(self.anno_path, delimiter=',')

        start_frames = {
            'bird1_1': 1, 'bird1_2': 775, 'bird1_3': 1573, 'car1_1': 1, 'car1_2': 751, 'car1_3': 1627,
            'car6_1': 1, 'car6_2': 487, 'car6_3': 1807, 'car6_4': 2953, 'car6_5': 3925,
            'car8_1': 1, 'car8_2': 1657, 'car16_1': 1, 'car16_2': 415,
            'group1_1': 1, 'group1_2': 1333, 'group1_3': 2515, 'group1_4': 3925,
            'group2_1': 1, 'group2_2': 907, 'group2_3': 1771,
            'group3_1': 1, 'group3_2': 1567, 'group3_3': 2827, 'group3_4': 4369,
            'person2_1': 1, 'person2_2': 1189, 'person4_1': 1, 'person4_2': 1501,
            'person5_1': 1, 'person5_2': 887, 'person7_1': 1, 'person7_2': 1249,
            'person8_1': 1, 'person8_2': 1075, 'person12_1': 1, 'person12_2': 601,
            'person14_1': 1, 'person14_2': 847, 'person14_3': 1813, 'person17_1': 1, 'person17_2': 1501,
            'person19_1': 1, 'person19_2': 1243, 'person19_3': 2791,
            'truck4_1': 1, 'truck4_2': 577, 'uav1_1': 1, 'uav1_2': 1555, 'uav1_3': 2473
        }

        if subset.lower() == 'uav20l':
            self.images_path = os.path.join(root, 'data_seq', 'UAV123', sequence)
            self.paths = sorted([filename for filename in os.listdir(self.images_path) if filename.endswith('.jpg')])
        else:
            if sequence in start_frames:
                tokens = sequence.split('_')
                self.images_path = os.path.join(root, 'data_seq', 'UAV123', tokens[0])
                self.paths = sorted(
                    [filename for filename in os.listdir(self.images_path) if filename.endswith('.jpg')])
                self.paths = self.paths[start_frames[sequence]-1:]
            else:
                self.images_path = os.path.join(root, 'data_seq', 'UAV123', sequence)
                self.paths = sorted(
                    [filename for filename in os.listdir(self.images_path) if filename.endswith('.jpg')])

        self.n_frames = min(len(self.paths), self.groundtruth.shape[0])
        self.groundtruth = self.groundtruth[:self.n_frames, :]

        self.preload = preload
        if preload:
            first_image = cv2.imread(os.path.join(self.images_path, self.paths[0]))
            if first_image.ndim == 2:
                first_image = cv2.cvtColor(first_image, cv2.COLOR_GRAY2RGB)
            elif first_image.ndim == 3:
                first_image = cv2.cvtColor(first_image, cv2.COLOR_BGR2RGB)

            h, w, c = first_image.shape
            images = np.zeros((self.n_frames, h, w, c))
            images[0, :, :, :] = first_image

            for idx, path in enumerate(self.paths[1:]):
                image = cv2.imread(os.path.join(self.images_path, self.paths[idx]))
                if image.ndim == 2:
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                elif image.ndim == 3:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                images[idx, :, :, :] = image
            self.images = images

    def __getitem__(self, index):
        if self.preload:
            img = self.images[index, :, :, :]
        else:
            img = cv2.imread(os.path.join(self.images_path, self.paths[index]))
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            elif img.ndim == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img, self.groundtruth[index, ...]

    def __len__(self):
        return self.n_frames

    def get_groundtruth(self):
        return self.groundtruth


class UAVDatasetReader(Dataset):
    def __init__(self, root, subset='uav123', preload=False):
        assert subset.lower() in ['uav123', 'uav20l'], "Subset must be one of ['uav123', 'uav20l']"
        self.root = root
        self.subset = subset
        print('Getting filelist...')
        self.videos = [filename.split('.')[0] for filename in os.listdir(os.path.join(root, 'anno', subset.upper()))
                       if filename.endswith('.txt')]
        print('List of videos: ', self.videos)
        # self.videos = self.videos[1:2]
        self.preload = preload

    def __getitem__(self, item):
        return self.videos[item], UAVVideoReader(self.root, self.subset.upper(), self.videos[item],
                                                 preload=self.preload)

    def __len__(self):
        return len(self.videos)


class VOTVideoReader(Dataset):
    def __init__(self, root, sequence):
        # assert subset in ['2014', '2015', '2016', '2017', 'cfnet-validation'], \
        #     "subset must be one of ['2014', '2015', '2016', '2017', 'cfnet-validation']"
        self.data_path = os.path.join(root, sequence)

        self.groundtruth = np.loadtxt(os.path.join(self.data_path, 'groundtruth.txt'), delimiter=',')
        self.name = sequence
        self.n_frames = self.groundtruth.shape[0]
        self.paths = sorted([img_path for img_path in os.listdir(self.data_path) if img_path.endswith('.jpg')])
        self.images = []
        for path in self.paths:
            img = cv2.imread(os.path.join(self.data_path, path))
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            elif img.ndim == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.images.append(img)
        self.total_time = 0

    def __getitem__(self, index):
        img = self.images[index]
        return img, _poly(np.asarray(self.groundtruth[index, ...]), center=False)

    def __len__(self):
        return self.n_frames

    def get_groundtruth(self):
        return np.asarray([_poly(np.asarray(gt), center=False) for gt in self.groundtruth])


class VOTDatasetReader(Dataset):
    def __init__(self, root, preload=False):
        self.root = root
        self.videos = get_video_names(root)
        self.preload = False
        if preload:
            print("Preloading entire dataset into memory...")
            self.video_readers = {self.videos[item]: VOTVideoReader(self.root, self.videos[item])
                                  for item in range(len(self.videos))}

    def __getitem__(self, item):
        if self.preload:
            return self.videos[item], self.video_readers[self.videos[item]]
        return self.videos[item], VOTVideoReader(self.root, self.videos[item])

    def __len__(self):
        return len(self.videos)
