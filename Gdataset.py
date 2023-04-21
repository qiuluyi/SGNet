from PIL import Image
import random
import cv2
import numpy as np
import torch
import torch.nn.init
import torch.optim.lr_scheduler
import torch.utils.data as data
from scipy import misc
from skimage import io
from preprocess import *

class boundary_dataset(torch.utils.data.Dataset):
    def __init__(self, ids, data_files=DATA_FOLDER, label_files=LABEL_FOLDER,
                 cache=False, augmentation=True):
        super(boundary_dataset, self).__init__()
        self.augmentation = augmentation
        self.cache = cache
        self.data_files = [DATA_FOLDER.format(id) for id in ids]
        self.label_files = [Boundary_FOLDER.format(id) for id in ids]
        self.data_cache_ = {}
        self.label_cache_ = {}

    def __len__(self):
        return 10000

    @classmethod
    def data_augmentation(cls, *arrays, flip=True, mirror=True):
        will_flip, will_mirror = False, False
        if flip and random.random() < 0.5:
            will_flip = True
        if mirror and random.random() < 0.5:
            will_mirror = True

        results = []
        for array in arrays:
            if will_flip:
                if len(array.shape) == 2:
                    array = array[::-1, :]
                else:
                    array = array[:, ::-1, :]
            if will_mirror:
                if len(array.shape) == 2:
                    array = array[:, ::-1]
                else:
                    array = array[:, :, ::-1]
            results.append(np.copy(array))
        return tuple(results)

    def __getitem__(self, i):
        random_idx = random.randint(0, len(self.data_files) - 1)
        if random_idx in self.data_cache_.keys():
            data = self.data_cache_[random_idx]
        else:
            data = 1 / 255 * np.asarray(io.imread(self.data_files[random_idx]).transpose((2, 0, 1)), dtype='float32')

            if self.cache:
                self.data_cache_[random_idx] = data
        if random_idx in self.label_cache_.keys():
            label = self.label_cache_[random_idx]
        else:
            label = io.imread(self.label_files[random_idx])
            label = Image.fromarray(label.astype('uint8')).convert('RGB')
            label = np.array(label)
            label = np.asarray(convert_from_color(label), dtype='int64')

            if self.cache:
                self.label_cache_[random_idx] = label

        x1, x2, y1, y2 = get_random_pos(data, WINDOW_SIZE)
        data_p = data[:, x1:x2, y1:y2]
        label_p = label[x1:x2, y1:y2]
        data_p, label_p = self.data_augmentation(data_p, label_p)
        return (torch.from_numpy(data_p), torch.from_numpy(label_p))
