import os
import torch
import torch.nn as nn
import torch.nn.init
import torch.optim.lr_scheduler
import torch.utils.data as data
import tqdm
from Sdataset import *
from models.DLinkNet import *
from models.LinkNet import *
from models.Unet_plus import *
from models.MAResUNet import *
from models.Deeplabv3 import *
from models.Deeplabv3_plus import *
from preprocess import *

try:
    from urllib.request import URLopener
except ImportError:
    from urllib import URLopener

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


def train(net, optimizer, epochs, scheduler=None, weights=WEIGHTS, save_epoch=10):
    losses = np.zeros(1000000)
    mean_losses = np.zeros(100000000)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.NLLLoss2d(weight=weights.to(device))
    iter_ = 0

    for e in range(1, epochs + 1):
        if scheduler is not None:
            scheduler.step()
        net.train()

        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            output = net(data)
            loss = CrossEntropy2d(output, target, weights.to(device))
            loss.backward()
            optimizer.step()
            losses[iter_] = loss.item()
            mean_losses[iter_] = np.mean(losses[max(0, iter_ - 100):iter_])

            if iter_ % 100 == 0:
                rgb = np.asarray(255 * np.transpose(data.data.cpu().numpy()[0], (1, 2, 0)), dtype='uint8')
                pred = np.argmax(output.data.cpu().numpy()[0], axis=0)
                gt = target.data.cpu().numpy()[0]

                print('SDbranch Train (epoch {}/{}) [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy: {}'.format(e, epochs,
                                                                                                          batch_idx,
                                                                                                          len(train_loader),
                                                                                                          100. * batch_idx / len(
                                                                                                              train_loader),
                                                                                                          loss.item(),
                                                                                                          accuracy(pred,
                                                                                                                   gt)))
                # plt.plot(mean_losses[:iter_]) and plt.show()
                # fig = plt.figure()
                # fig.add_subplot(131)
                # plt.imshow(rgb)
                # plt.title('RGB')
                # fig.add_subplot(132)
                # plt.imshow(convert_to_color(gt))
                # plt.title('Ground truth')
            # fig.add_subplot(133)
            # plt.title('Prediction')
            # plt.imshow(convert_to_color(pred))
            # plt.show()
            iter_ += 1

            del (data, target, loss)

        if e % save_epoch == 0:
            # acc = test(net, valid_ids, all=False, stride=min(WINDOW_SIZE))
            
            if os.path.exists('./linknet') is False:
                os.mkdir('./linknet')
            
            torch.save(net.state_dict(), './linknet/linknet_{}'.format(e))
    torch.save(net.state_dict(), './linknet/linknet_final')

def train_method():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = LinkNet()
    # net = resnet101(n_class=2, output_stride=16, pretrained=True) # Deeplab v3
    # net = DeepLabV3Plus(nclass=2)
    # net = DinkNet34()
    # net = DinkNet50()
    # net = DinkNet101()
    # net = UnetPlusPlus(num_classes=2, deep_supervision=False)
    # net = MAResUNet()

    vgg_url = 'https://download.pytorch.org/vgg16_bn-6c64b313.pth'
    if not os.path.isfile('pretrain/vgg16_bn-6c64b313.pth'):
        weights = URLopener().retrieve(vgg_url, 'pretrain/vgg16_bn-6c64b313.pth')
    vgg_weights = torch.load('pretrain/vgg16_bn-6c64b313.pth')
    mapped_weights = {}
    for k_vgg, k_segnet in zip(vgg_weights.keys(), net.state_dict().keys()):
        if "features" in k_vgg:
            mapped_weights[k_segnet] = vgg_weights[k_vgg]
    try:
        net.load_state_dict(mapped_weights)
    except:
        pass

    net.to(device)
    base_lr = 0.01
    params_dict = dict(net.named_parameters())
    params = []
    for key, value in params_dict.items():
        if '_D' in key:
            params += [{'params': [value], 'lr': base_lr}]
        else:
            params += [{'params': [value], 'lr': base_lr / 2}]
    optimizer = torch.optim.SGD(net.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [25, 35, 45], gamma=0.1)
    train(net, optimizer, 100, scheduler)



if __name__ == "__main__":
    train_ids = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15']
    train_set = Sdataset(train_ids, cache=CACHE)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE)

    valid_ids = ['16', '17', '18']
    valid_set = Sdataset(train_ids, cache=CACHE)
    valid_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE)

    test_ids = ['19', '20']
    test_set = Sdataset(test_ids, cache=CACHE)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE)

    train_method()
