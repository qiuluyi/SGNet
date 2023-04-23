# Dataset class
import random
# imports and stuff
import numpy as np
# Matplotlib
# Torch imports
import torch.nn as nn
import torch.nn.init
import torch
import torch.nn.init
import torch.optim.lr_scheduler
import torch.utils.data as data
from scipy import misc
from skimage import io
from PIL import Image
from preprocess import *

# Dataset class
class road_dataset(torch.utils.data.Dataset):
    def __init__(self, ids, data_files=DATA_FOLDER, boundary_files=Boundary_FOLDER, label_files=LABEL_FOLDER,
                 cache=False, augmentation=True):
        super(road_dataset, self).__init__()

        self.augmentation = augmentation
        self.cache = cache

        # List of files
        # 原始IRRG图像
        # self.data_files = [Predict_FOLDER.format(id) for id in ids]
        self.data_files = [DATA_FOLDER.format(id) for id in ids]
        self.boundary_files = [Boundary_FOLDER.format(id) for id in ids]
        self.label_files = [LABEL_FOLDER.format(id) for id in ids]


        # Sanity check : raise an error if some files do not exist
        # for f in self.data_files + self.label_files:
        #     if not os.path.isfile(f):
        #         raise KeyError('{} is not a file !'.format(f))

        # Initialize cache dicts
        self.data_cache_ = {}
        self.label_cache_ = {}
        self.boundary_cache_={}


    def __len__(self):
        # Default epoch size is 10000 samples
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
        # random_idx = random.randint(0, len(self.img_files) - 1)

        if random_idx in self.data_cache_.keys():
            data = self.data_cache_[random_idx]
        else:
            #   Data is normalized in [0, 1]
            data = 1 / 255 * np.asarray(io.imread(self.data_files[random_idx]).transpose((2, 0, 1)), dtype='float32')


            if self.cache:
                self.data_cache_[random_idx] = data
                # self.img_cache_[random_idx] = data

        if random_idx in self.label_cache_.keys():
            label = self.label_cache_[random_idx]
            boundary = self.boundary_cache_[random_idx]
            
        else:
            boundary = Image.open(self.boundary_files[random_idx])
            boundary = np.asarray(boundary.convert("RGB"))
            boundary =  np.asarray(convert_from_color(boundary), dtype='int64')
            label = io.imread(self.label_files[random_idx])
            label = np.asarray(convert_from_color(label), dtype='int64')

            if self.cache:
                self.label_cache_[random_idx] = label
                self.boundary_cache_[random_idx] = boundary

        # Get a random patch
        x1, x2, y1, y2 = get_random_pos(data, WINDOW_SIZE)
        data_p = data[:, x1:x2, y1:y2]
        boundary_p = boundary[x1:x2, y1:y2]
        label_p = label[x1:x2, y1:y2]
        # Data augmentation
        data_p,boundary_p, label_p = self.data_augmentation(data_p,boundary_p, label_p)

        # Return the torch.Tensor values
        return (torch.from_numpy(data_p),torch.from_numpy(boundary_p), torch.from_numpy(label_p))
