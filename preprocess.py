import itertools
import random
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn.init
import torch.optim.lr_scheduler
from sklearn.metrics import confusion_matrix

WINDOW_SIZE = (256, 256)  # seg size
STRIDE = 32  # Stride for testing
IN_CHANNELS = 3
FOLDER = "/home/qly/sdbranch/"  # Replace with your "/path/to/the/ISPRS/dataset/folder/"
BATCH_SIZE = 5  # Number of samples in a mini-batch
LABELS = ["roads"]  # Label name
N_CLASSES = 2  # Number of classes
WEIGHTS = torch.ones(N_CLASSES)  # Weights for class balancing
CACHE = True  # Store the dataset in-memory

DATASET = 'Ottawa'
if DATASET == 'Ottawa':
    MAIN_FOLDER = FOLDER + 'Ottawa-Dataset/'
    DATA_FOLDER = MAIN_FOLDER + 'image/Ottawa-{}.tif'
    Boundary_FOLDER = MAIN_FOLDER + 'boundary/Ottawa-{}.png'
    LABEL_FOLDER = MAIN_FOLDER + 'label/Ottawa-{}.png'

palette = {0: (255, 255, 255),  # target (white)
           1: (0, 0, 0) }  # others (black)

invert_palette = {v: k for k, v in palette.items()}

def convert_to_color(arr_2d, palette=palette):
    arr_3d = np.zeros((arr_2d.shape[0], arr_2d.shape[1], 3), dtype=np.uint8)

    for c, i in palette.items():
        m = arr_2d == c
        arr_3d[m] = i

    return arr_3d

def convert_from_color(arr_3d, palette=invert_palette):
    """ RGB-color encoding to grayscale labels """
    arr_2d = np.zeros((arr_3d.shape[0], arr_3d.shape[1]), dtype=np.uint8)

    for c, i in palette.items():
        m = np.all(arr_3d == np.array(c).reshape(1, 1, 3), axis=2)
        arr_2d[m] = i

    return arr_2d

def get_random_pos(img, window_shape):
    """ Extract of 2D random patch of shape window_shape in the image """
    w, h = window_shape
    W, H = img.shape[-2:]
    x1 = random.randint(0, W - w - 1)
    x2 = x1 + w
    y1 = random.randint(0, H - h - 1)
    y2 = y1 + h
    return x1, x2, y1, y2

def CrossEntropy2d(input, target, weight=None, size_average=True):
    dim = input.dim()
    if dim == 2:
        return F.cross_entropy(input, target, weight, size_average)
    elif dim == 4:
        output = input.view(input.size(0), input.size(1), -1)
        output = torch.transpose(output, 1, 2).contiguous()
        output = output.view(-1, output.size(2))
        target = target.view(-1)
        return F.cross_entropy(output, target, weight, size_average)
    else:
        raise ValueError('Expected 2 or 4 dimensions (got {})'.format(dim))

def accuracy(input, target):
    return 100 * float(np.count_nonzero(input == target)) / target.size

def sliding_window(top, step=10, window_size=(20, 20)):
    for x in range(0, top.shape[0], step):
        if x + window_size[0] > top.shape[0]:
            x = top.shape[0] - window_size[0]
        for y in range(0, top.shape[1], step):
            if y + window_size[1] > top.shape[1]:
                y = top.shape[1] - window_size[1]
            yield x, y, window_size[0], window_size[1]

def count_sliding_window(top, step=10, window_size=(20, 20)):
    c = 0
    for x in range(0, top.shape[0], step):
        if x + window_size[0] > top.shape[0]:
            x = top.shape[0] - window_size[0]
        for y in range(0, top.shape[1], step):
            if y + window_size[1] > top.shape[1]:
                y = top.shape[1] - window_size[1]
            c += 1
    return c

def grouper(n, iterable):
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, n))
        if not chunk:
            return
        yield chunk
