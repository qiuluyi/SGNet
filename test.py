import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init
import torch.optim.lr_scheduler
import torch.utils.data as data
import tqdm
from torch.autograd import Variable
from dataset import *
from models.SGNet import *
from models.DLinkNet import *
from models.LinkNet import *
from models.Unet_plus import *
from models.MAResUNet import *
from models.Deeplabv3 import *
from models.Deeplabv3_plus import *
import os
import yimage
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix
import tqdm
from skimage import io
from collections import Counter
from preprocess import *
from PIL import Image

def cal(confu_mat_total,file_name, save_path='./'):
    class_num = confu_mat_total.shape[0]
    confu_mat = confu_mat_total.astype(np.float32) + 0.0001
    col_sum = np.sum(confu_mat, axis=1) 
    raw_sum = np.sum(confu_mat, axis=0)

    oa = 0
    for i in range(class_num):
        oa = oa + confu_mat[i, i]
    oa = oa / confu_mat.sum()

    '''Kappa'''
    pe_fz = 0
    for i in range(class_num):
        pe_fz += col_sum[i] * raw_sum[i]
    pe = pe_fz / (np.sum(confu_mat) * np.sum(confu_mat))
    kappa = (oa - pe) / (1 - pe)

    TP = [] 

    for i in range(class_num):
        TP.append(confu_mat[i, i])

    TP = np.array(TP)
    FN = col_sum - TP
    FP = raw_sum - TP

    f1_m = []
    iou_m = []
    for i in range(class_num):
        # 写出f1-score
        f1 = TP[i] * 2 / (TP[i] * 2 + FP[i] + FN[i])
        f1_m.append(f1)
        iou = TP[i] / (TP[i] + FP[i] + FN[i])
        iou_m.append(iou)

    f1_m = np.array(f1_m)
    iou_m = np.array(iou_m)
    if save_path is not None:
        with open(save_path + file_name+'.txt', 'w') as f:
            f.write('OA:\t%.4f\n' % (oa * 100))
            f.write('kappa:\t%.4f\n' % (kappa * 100))
            f.write('mf1-score:\t%.4f\n' % (np.mean(f1_m) * 100))
            f.write('mIou:\t%.4f\n' % (np.mean(iou_m) * 100))

            # 写出precision
            f.write('precision:\n')
            for i in range(class_num):
                f.write('%.4f\t' % (float(TP[i] / raw_sum[i]) * 100))
            f.write('\n')

            # 写出recall
            f.write('recall:\n')
            for i in range(class_num):
                f.write('%.4f\t' % (float(TP[i] / col_sum[i]) * 100))
            f.write('\n')

            # 写出f1-score
            f.write('f1-score:\n')
            for i in range(class_num):
                f.write('%.4f\t' % (float(f1_m[i]) * 100))
            f.write('\n')

            # 写出 IOU
            f.write('Iou:\n')
            for i in range(class_num):
                f.write('%.4f\t' % (float(iou_m[i]) * 100))
            f.write('\n')

def cal_acc(pred, gt):
    pixel_num = pred.shape[0] * pred.shape[1]
    pred[gt == 0] = 0
    boundary_num = np.sum(gt == 0)
    true_num = np.sum(gt == pred)
    pixel_acc = (true_num - boundary_num) / (pixel_num - boundary_num)
    return pixel_acc, true_num - boundary_num, pixel_num - boundary_num

def cal_f1(pred, gt):
    pred[gt == 0] = 0
    f1 = f1_score(gt.flatten(), pred.flatten(), average=None)
    cm = confusion_matrix(gt.flatten(), pred.flatten(), labels=[i for i in range(2)])
    return f1, cm

def metrics(pred_path,label_path):
    gts = sorted(os.listdir(label_path))
    pred_num = sorted(os.listdir(pred_path))
    preds=os.listdir(label_path)
    for num in pred_num:
        up_all = []
        down_all = []
        cm_init = np.zeros((2, 2))
        for name in tqdm.tqdm(preds):
            print("******************{}******************".format(name))
            pred = yimage.io.read_image(os.path.join(pred_path, name))
            gt = yimage.io.read_image(os.path.join(label_path, name))
            pred = convert_from_color(pred)
            gt = convert_from_color(gt)
            gt = gt + 1
            gt = np.where(gt == 2, 0, gt)
            acc, up, down = cal_acc(pred, gt)
            f1, cm = cal_f1(pred, gt)
            up_all.append(up)
            down_all.append(down)
            cm_init += cm
            print('The accuracy of the {} is {}\n, and the f1 score is {}'.format(name, acc, f1))
        cal(cm_init,file_name=num)
        acc_all = np.sum(up_all) / np.sum(down_all)
        print('The OA is {}'.format(acc_all))
    return acc_all


def test(net, test_ids, all=False, stride=WINDOW_SIZE[0], batch_size=BATCH_SIZE, window_size=WINDOW_SIZE):
    # Use the network on the test set
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_images = (1 / 255 * np.asarray(io.imread(), dtype='float32') for id in test_ids)
    test_labels = (np.asarray(io.imread(LABEL_FOLDER.format(id)), dtype='uint8') for id in test_ids)
    num = 0
    all_preds = []
    all_gts = []

    net.cuda()
    net.train()
    net.eval()

    for img, gt in  tqdm.tqdm(zip(test_images, test_labels), total=len(test_ids), leave=False):
        pred = np.zeros(img.shape[:2] + (N_CLASSES,))
        total = count_sliding_window(img, step=stride, window_size=window_size) // batch_size

        for i, coords in enumerate(
                tqdm.tqdm(grouper(batch_size, sliding_window(img, step=stride, window_size=window_size)), total=total,
                     leave=False)):

            # Display in progress results
            if i > 0 and total > 10 and i % int(10 * total / 100) == 0:
                _pred = np.argmax(pred, axis=-1)
                # fig = plt.figure()
                # fig.add_subplot(1, 3, 1)
                # plt.imshow(np.asarray(255 * img, dtype='uint8'))
                # fig.add_subplot(1, 3, 2)
                # plt.imshow(convert_to_color(_pred))
                # fig.add_subplot(1, 3, 3)
                # plt.imshow(gt)
                # clear_output()
                # plt.show()

            # Build the tensor
            image_patches = [np.copy(img[x:x + w, y:y + h]).transpose((2, 0, 1)) for x, y, w, h in coords]
            image_patches = np.asarray(image_patches)
            image_patches = Variable(torch.from_numpy(image_patches).cuda(), volatile=True)

            # Do the inference
            predtion = net(image_patches)
            outs = predtion.data.cpu().numpy()

            # s_feature, b1_feature, b2_feature, b3_feature, b4_feature, b5_feature, b_feature, refined_depth=net(image_patches)
            # outs = refined_depth.data.cpu().numpy()

            # Fill in the results array
            for out, (x, y, w, h) in zip(outs, coords):
                out = out.transpose((1, 2, 0))
                pred[x:x + w, y:y + h] += out

            del (outs)

        pred = np.argmax(pred, axis=-1)
        # Display the result
        # clear_output()
        # fig = plt.figure()
        # fig.add_subplot(1, 3, 1)
        # plt.imshow(np.asarray(255 * img, dtype='uint8'))
        # fig.add_subplot(1, 3, 2)
        # plt.imshow(convert_to_color(pred))
        # fig.add_subplot(1, 3, 3)
        # plt.imshow(gt)
        # plt.show()
        pre_img = convert_to_color(pred)
        im = Image.fromarray(pre_img)
        print("test_id", test_ids[num])
        pred_path='./DemoNet/'
        if os.path.exists(pred_path) is False:
            os.mkdir(pred_path)

        if os.path.exists(pred_path) is False:
            os.mkdir(pred_path)
        img=os.path.join(pred_path,'Ottawa-{}.png'.format(test_ids[num]))
        im.save(img)
        num = num + 1
        all_preds.append(pred)
        all_gts.append(gt)

    accuracy = metrics(pred_path,LABEL_FOLDER)

    if all:
        return accuracy, all_preds, all_gts
    else:
        return accuracy


def test_method(test_ids):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = LinkNet()
    # net = resnet101(n_class=2, output_stride=16, pretrained=True) # Deeplab v3
    # net = DeepLabV3Plus(nclass=2)
    # net = DinkNet34()
    # net = DinkNet50()
    # net = DinkNet101()
    # net = UnetPlusPlus(num_classes=2, deep_supervision=False)
    # net = MAResUNet()
    # net = SGNet()

    net.to(device)
    checkpoint = torch.load('./Demo/Demo_epochx')
    net.load_state_dict(checkpoint)
    acc = test(net, test_ids, all=False, stride=min(WINDOW_SIZE))
    return acc


if __name__ == "__main__":
    train_ids = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15']
    valid_id = ['16', '17', '18']
    train_set = road_dataset(train_ids, cache=CACHE)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE)

    test_ids = ['19', '20']
    test_set = road_dataset(test_ids, cache=CACHE)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE)

    test_method(test_ids)
