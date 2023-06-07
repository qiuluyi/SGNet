import torch
import torch.nn as nn
import torch.nn.init
import torch.optim.lr_scheduler
import getopt
import numpy
import PIL
import PIL.Image
import sys
import torch
import math
from preprocess import *

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, outsize, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(outsize)
        self.max_pool = nn.AdaptiveMaxPool2d(outsize)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class ASPP(nn.Module):
    def __init__(self, in_channel=512, depth=256, stride=1):
        super(ASPP, self).__init__()
        self.mean = nn.AdaptiveMaxPool2d((stride, stride))
        self.conv = nn.Conv2d(in_channel, depth, 1, 1)
        self.atrous_block1 = nn.Conv2d(in_channel, depth, 1, stride)
        self.atrous_block6 = nn.Conv2d(in_channel, depth, 3, stride, padding=6, dilation=6)
        self.atrous_block12 = nn.Conv2d(in_channel, depth, 3, stride, padding=12, dilation=12)
        self.atrous_block18 = nn.Conv2d(in_channel, depth, 3, stride, padding=18, dilation=18)
        self.conv_1x1_output = nn.Conv2d(depth * 5, depth, 1, stride)

    def forward(self, x):
        size = x.shape[2:]
        image_features = self.mean(x)
        image_features = self.conv(image_features)
        image_features = F.upsample(image_features, size=size, mode='bilinear')
        atrous_block1 = self.atrous_block1(x)
        atrous_block6 = self.atrous_block6(x)
        atrous_block12 = self.atrous_block12(x)
        atrous_block18 = self.atrous_block18(x)
        net = self.conv_1x1_output(torch.cat([image_features, atrous_block1, atrous_block6,
                                              atrous_block12, atrous_block18], dim=1))
        return net

class SDbranch(nn.Module):
    def weight_init(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_normal(m.weight.data)

    def __init__(self, in_channels=IN_CHANNELS, out_channels=N_CLASSES):
        super(SDbranch, self).__init__()
        self.pool = nn.MaxPool2d(2, return_indices=True)
        self.unpool = nn.MaxUnpool2d(2)

        self.conv1_1 = nn.Conv2d(in_channels, 64, 3, padding=1)
        self.conv1_1_bn = nn.BatchNorm2d(64)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv1_2_bn = nn.BatchNorm2d(64)

        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv2_1_bn = nn.BatchNorm2d(128)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv2_2_bn = nn.BatchNorm2d(128)

        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv3_1_bn = nn.BatchNorm2d(256)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3_2_bn = nn.BatchNorm2d(256)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3_3_bn = nn.BatchNorm2d(256)

        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv4_1_bn = nn.BatchNorm2d(512)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv4_2_bn = nn.BatchNorm2d(512)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv4_3_bn = nn.BatchNorm2d(512)
        self.aspp4 = ASPP(in_channel=512, depth=512, stride=2)

        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_1_bn = nn.BatchNorm2d(512)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_2_bn = nn.BatchNorm2d(512)
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_3_bn = nn.BatchNorm2d(512)

        self.aspp = ASPP(in_channel=512, depth=512)
        self.cbam5 = ChannelAttention(in_planes=512, outsize=16)
        self.conv5_3_D = nn.Conv2d(1024, 512, 3, padding=1)
        self.conv5_3_D_bn = nn.BatchNorm2d(512)
        self.conv5_3_D_1 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_3_D_1_bn = nn.BatchNorm2d(512)
        self.conv5_2_D = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_2_D_bn = nn.BatchNorm2d(512)
        self.conv5_1_D = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_1_D_bn = nn.BatchNorm2d(512)

        self.cbam4 = ChannelAttention(in_planes=512, outsize=32)
        self.conv4_3_D = nn.Conv2d(1024, 512, 3, padding=1)
        self.conv4_3_D_bn = nn.BatchNorm2d(512)
        self.conv4_3_D_1 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv4_3_D_1_bn = nn.BatchNorm2d(512)
        self.conv4_2_D = nn.Conv2d(512, 512, 3, padding=1)
        self.conv4_2_D_bn = nn.BatchNorm2d(512)
        self.conv4_1_D = nn.Conv2d(512, 256, 3, padding=1)
        self.conv4_1_D_bn = nn.BatchNorm2d(256)

        self.cbam3 = ChannelAttention(in_planes=256, outsize=64)
        self.conv3_3_D = nn.Conv2d(512, 256, 3, padding=1)
        self.conv3_3_D_bn = nn.BatchNorm2d(256)
        self.conv3_3_D_1 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3_3_D_1_bn = nn.BatchNorm2d(256)
        self.conv3_2_D = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3_2_D_bn = nn.BatchNorm2d(256)
        self.conv3_1_D = nn.Conv2d(256, 128, 3, padding=1)
        self.conv3_1_D_bn = nn.BatchNorm2d(128)

        self.cbam2 = ChannelAttention(in_planes=128, outsize=128)
        self.conv2_2_D = nn.Conv2d(256, 128, 3, padding=1)
        self.conv2_2_D_bn = nn.BatchNorm2d(128)
        self.conv2_2_D_1 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv2_2_D_1_bn = nn.BatchNorm2d(128)
        self.conv2_1_D = nn.Conv2d(128, 64, 3, padding=1)
        self.conv2_1_D_bn = nn.BatchNorm2d(64)

        self.cbam1 = ChannelAttention(in_planes=64, outsize=256)
        self.conv1_2_D = nn.Conv2d(128, 64, 3, padding=1)
        self.conv1_2_D_bn = nn.BatchNorm2d(64)
        self.conv1_2_D_1 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv1_2_D_1_bn = nn.BatchNorm2d(64)
        self.conv1_1_D = nn.Conv2d(64, 2, 3, padding=1)

    def forward(self, x):
        x1 = self.conv1_1_bn(F.leaky_relu(self.conv1_1(x)))
        x1 = self.conv1_2_bn(F.leaky_relu(self.conv1_2(x1)))
        downx1, mask1 = self.pool(x1)

        x2 = self.conv2_1_bn(F.leaky_relu(self.conv2_1(downx1)))
        x2 = self.conv2_2_bn(F.leaky_relu(self.conv2_2(x2)))
        downx2, mask2 = self.pool(x2)

        x3 = self.conv3_1_bn(F.leaky_relu(self.conv3_1(downx2)))
        x3 = self.conv3_2_bn(F.leaky_relu(self.conv3_2(x3)))
        x3 = self.conv3_3_bn(F.leaky_relu(self.conv3_3(x3)))
        downx3, mask3 = self.pool(x3)

        x4 = self.conv4_1_bn(F.leaky_relu(self.conv4_1(downx3)))
        x4 = self.conv4_2_bn(F.leaky_relu(self.conv4_2(x4)))
        x4 = self.conv4_3_bn(F.leaky_relu(self.conv4_3(x4)))
        downx4, mask4 = self.pool(x4)

        x5 = self.conv5_1_bn(F.leaky_relu(self.conv5_1(downx4)))
        x5 = self.conv5_2_bn(F.leaky_relu(self.conv5_2(x5)))
        x5 = self.conv5_3_bn(F.leaky_relu(self.conv5_3(x5)))
        downx5, mask5 = self.pool(x5)
        aspp = self.aspp(downx5)

        dx5 = self.unpool(aspp, mask5)
        cbam5 = self.cbam5(x5)
        dx5 = torch.cat([dx5, cbam5], dim=1)
        dx5 = self.conv5_3_D_bn(F.leaky_relu(self.conv5_3_D(dx5)))
        dx5 = self.conv5_3_D_1_bn(F.leaky_relu(self.conv5_3_D_1(dx5)))
        dx5 = self.conv5_2_D_bn(F.leaky_relu(self.conv5_2_D(dx5)))
        dx5 = self.conv5_1_D_bn(F.leaky_relu(self.conv5_1_D(dx5)))

        dx4 = self.unpool(dx5, mask4)
        cbam4 = self.cbam4(x4)
        dx4 = torch.cat([dx4, cbam4], dim=1)
        dx4 = self.conv4_3_D_bn(F.leaky_relu(self.conv4_3_D(dx4)))
        dx4 = self.conv4_3_D_1_bn(F.leaky_relu(self.conv4_3_D_1(dx4)))
        dx4 = self.conv4_2_D_bn(F.leaky_relu(self.conv4_2_D(dx4)))
        dx4 = self.conv4_1_D_bn(F.leaky_relu(self.conv4_1_D(dx4)))

        dx3 = self.unpool(dx4, mask3)
        cbam3 = self.cbam3(x3)
        dx3 = torch.cat([dx3, cbam3], dim=1)
        dx3 = self.conv3_3_D_bn(F.leaky_relu(self.conv3_3_D(dx3)))
        dx3 = self.conv3_3_D_1_bn(F.leaky_relu(self.conv3_3_D_1(dx3)))
        dx3 = self.conv3_2_D_bn(F.leaky_relu(self.conv3_2_D(dx3)))
        dx3 = self.conv3_1_D_bn(F.leaky_relu(self.conv3_1_D(dx3)))

        dx2 = self.unpool(dx3, mask2)
        cbam2 = self.cbam2(x2)
        dx2 = torch.cat([dx2, cbam2], dim=1)
        dx2 = self.conv2_2_D_bn(F.leaky_relu(self.conv2_2_D(dx2)))
        dx2 = self.conv2_2_D_1_bn(F.leaky_relu(self.conv2_2_D_1(dx2)))
        dx2 = self.conv2_1_D_bn(F.leaky_relu(self.conv2_1_D(dx2)))

        dx1 = self.unpool(dx2, mask1)
        cbam1 = self.cbam1(x1)
        dx1 = torch.cat([dx1, cbam1], dim=1)
        dx1 = self.conv1_2_D_bn(F.leaky_relu(self.conv1_2_D(dx1)))
        dx1 = self.conv1_2_D_1_bn(F.leaky_relu(self.conv1_2_D_1(dx1)))
        dx1 = self.conv1_1_D(dx1)
        return dx1

class GDbranch(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.netVggOne = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )

        self.netVggTwo = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )

        self.netVggThr = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )

        self.netVggFou = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )

        self.netVggFiv = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )

        self.netScoreOne = torch.nn.Conv2d(in_channels=64, out_channels=2, kernel_size=1, stride=1, padding=0)
        self.netScoreTwo = torch.nn.Conv2d(in_channels=128, out_channels=2, kernel_size=1, stride=1, padding=0)
        self.netScoreThr = torch.nn.Conv2d(in_channels=256, out_channels=2, kernel_size=1, stride=1, padding=0)
        self.netScoreFou = torch.nn.Conv2d(in_channels=512, out_channels=2, kernel_size=1, stride=1, padding=0)
        self.netScoreFiv = torch.nn.Conv2d(in_channels=512, out_channels=2, kernel_size=1, stride=1, padding=0)

        self.netCombine = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=10, out_channels=2, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, tenInput):
        tenInput = tenInput * 255.0
        tenInput = tenInput - torch.tensor(data=[104.00698793, 116.66876762, 122.67891434], dtype=tenInput.dtype, device=tenInput.device).view(1, 3, 1, 1)

        tenVggOne = self.netVggOne(tenInput)
        tenVggTwo = self.netVggTwo(tenVggOne)
        tenVggThr = self.netVggThr(tenVggTwo)
        tenVggFou = self.netVggFou(tenVggThr)
        tenVggFiv = self.netVggFiv(tenVggFou)

        tenScoreOne = self.netScoreOne(tenVggOne)
        tenScoreTwo = self.netScoreTwo(tenVggTwo)
        tenScoreThr = self.netScoreThr(tenVggThr)
        tenScoreFou = self.netScoreFou(tenVggFou)
        tenScoreFiv = self.netScoreFiv(tenVggFiv)

        tenScoreOne = torch.nn.functional.interpolate(input=tenScoreOne, size=(tenInput.shape[2], tenInput.shape[3]), mode='bilinear', align_corners=False)
        tenScoreTwo = torch.nn.functional.interpolate(input=tenScoreTwo, size=(tenInput.shape[2], tenInput.shape[3]), mode='bilinear', align_corners=False)
        tenScoreThr = torch.nn.functional.interpolate(input=tenScoreThr, size=(tenInput.shape[2], tenInput.shape[3]), mode='bilinear', align_corners=False)
        tenScoreFou = torch.nn.functional.interpolate(input=tenScoreFou, size=(tenInput.shape[2], tenInput.shape[3]), mode='bilinear', align_corners=False)
        tenScoreFiv = torch.nn.functional.interpolate(input=tenScoreFiv, size=(tenInput.shape[2], tenInput.shape[3]), mode='bilinear', align_corners=False)
        tenScoreSix = self.netCombine(torch.cat([tenScoreOne, tenScoreTwo, tenScoreThr, tenScoreFou, tenScoreFiv], 1))

        return  tenScoreOne,tenScoreTwo, tenScoreThr,tenScoreFou,tenScoreFiv,tenScoreSix

def kernel_trans(kernel, weight):
    kernel_size = int(math.sqrt(kernel.size()[1]))
    kernel = F.conv2d(kernel, weight, stride=1, padding=int((kernel_size-1)/2))
    return kernel

def convbn(in_channels, out_channels, kernel_size=3,stride=1, padding=1):
    return nn.Sequential(
		nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
		nn.BatchNorm2d(out_channels)
	)

def weights_init(m):
    # Initialize filters with Gaussian random weights
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.ConvTranspose2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

class CSPNGenerateAccelerate(nn.Module):
    def __init__(self, in_channels, kernel_size):
        super(CSPNGenerateAccelerate, self).__init__()
        self.kernel_size = kernel_size
        self.generate = convbn(in_channels, self.kernel_size * self.kernel_size - 1, kernel_size=3, stride=1, padding=1)

    def forward(self, feature):
        guide = self.generate(feature)
        guide_sum = torch.sum(guide.abs(), dim=1).unsqueeze(1)
        guide = torch.div(guide, guide_sum)
        guide_mid = (1 - torch.sum(guide, dim=1)).unsqueeze(1)
        half1, half2 = torch.chunk(guide, 2, dim=1)
        output =  torch.cat((half1, guide_mid, half2), dim=1)
        return output

class CSPNAccelerate(nn.Module):
    def __init__(self, kernel_size, dilation=1, padding=1, stride=1):
        super(CSPNAccelerate, self).__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = padding
        self.stride = stride
        self.conv1= convbn(kernel_size*kernel_size*3,kernel_size*kernel_size)

    def forward(self, kernel, input, input0):
        bs = input.size()[0]
        c = input.size()[1]
        h, w = input.size()[2], input.size()[3]
        input0 = input0.reshape(bs, c, h * w)
        input_im2col = F.unfold(input, self.kernel_size, self.dilation, self.padding, self.stride)
        kernel = kernel.reshape(bs, self.kernel_size * self.kernel_size, h * w)
        mid_index = int((self.kernel_size*self.kernel_size-1)/2)
        input_im2col[:, mid_index:mid_index+3, :] = input0
        input_im2col= input_im2col.reshape(bs, input_im2col.size()[1], h ,w)
        input_im2col=self.conv1(input_im2col).reshape(bs, self.kernel_size * self.kernel_size, h * w)
        output = torch.einsum('ijk,ijk->ik', (input_im2col, kernel))
        return output.view(bs, 1, h, w)


class SGNet(nn.Module):
    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_normal(m.weight.data)

    def __init__(self, in_channels=IN_CHANNELS, out_channels=N_CLASSES):
        super(SGNet, self).__init__()
        # SDbranch
        self.sdbranch=SDbranch()
        # GDbranch
        self.gdbranch= GDbranch()
        
    def forward(self, x):
        # GDbranch
        b1_feature, b2_feature, b3_feature, b4_feature, b5_feature, b_feature = self.gdbranch(x)
        # SDbranch
        s_feature =  self.sdbranch(x)
        return s_feature, b1_feature, b2_feature, b3_feature, b4_feature, b5_feature, b_feature

class FRM(nn.Module):
    def __init__(self):
        super(FRM, self).__init__()

        self.backbone = SGNet()
        self.conv1=convbn(5, 3)
        self.mask_layer = convbn(3, 3)
        self.kernel_conf_layer = convbn(3, 3)
        self.iter_conf_layer = convbn(3, 12)
        self.iter_guide_layer3 = CSPNGenerateAccelerate(3, 3)
        self.iter_guide_layer5 = CSPNGenerateAccelerate(3, 5)
        self.iter_guide_layer7 = CSPNGenerateAccelerate(3, 7)
        self.softmax = nn.Softmax(dim=1)
        self.CSPN3 = CSPNAccelerate(3)
        self.CSPN5 = CSPNAccelerate(5, padding=2)
        self.CSPN7 = CSPNAccelerate(7, padding=3)
        self.cbam = ChannelAttention(in_planes=3, outsize=256,ratio=2)
        self.cbam_conv=convbn(3, 2)
        ks = 3
        encoder3 = torch.zeros(ks * ks, ks * ks, ks, ks).cuda()
        kernel_range_list = [i for i in range(ks - 1, -1, -1)]
        ls = []
        for i in range(ks):
            ls.extend(kernel_range_list)
        index = [[j for j in range(ks * ks - 1, -1, -1)], [j for j in range(ks * ks)], \
                 [val for val in kernel_range_list for j in range(ks)], ls]
        encoder3[index] = 1
        self.encoder3 = nn.Parameter(encoder3, requires_grad=False)

        ks = 5
        encoder5 = torch.zeros(ks * ks, ks * ks, ks, ks).cuda()
        kernel_range_list = [i for i in range(ks - 1, -1, -1)]
        ls = []
        for i in range(ks):
            ls.extend(kernel_range_list)
        index = [[j for j in range(ks * ks - 1, -1, -1)], [j for j in range(ks * ks)], \
                 [val for val in kernel_range_list for j in range(ks)], ls]
        encoder5[index] = 1
        self.encoder5 = nn.Parameter(encoder5, requires_grad=False)

        ks = 7
        encoder7 = torch.zeros(ks * ks, ks * ks, ks, ks).cuda()
        kernel_range_list = [i for i in range(ks - 1, -1, -1)]
        ls = []
        for i in range(ks):
            ls.extend(kernel_range_list)
        index = [[j for j in range(ks * ks - 1, -1, -1)], [j for j in range(ks * ks)], \
                 [val for val in kernel_range_list for j in range(ks)], ls]
        encoder7[index] = 1
        self.encoder7 = nn.Parameter(encoder7, requires_grad=False)

        weights_init(self)
        

    def forward(self, input):

        s_feature,  b1_feature, b2_feature, b3_feature, b4_feature, b5_feature,b_feature = self.backbone(input)
        s_depth = s_feature[:, 0:1, :, :]
        s_conf = s_feature[:, 1:2, :, :]
        b_depth, b_conf = torch.chunk(b_feature, 2, dim=1)
        s_conf, b_conf = torch.chunk(self.softmax(torch.cat((s_conf, b_conf), dim=1)), 2, dim=1)
        feature = s_depth * s_depth + b_conf * b_conf
        feature = self.conv1(torch.cat([feature,s_feature,b_feature],dim=1))
        d = feature
        valid_mask = torch.where(d > 0, torch.full_like(d, 1.0), torch.full_like(d, 0.0))
        
        # mask = self.mask_layer(feature)
        mask = torch.sigmoid(feature)
        mask = mask*valid_mask
        mask3 = mask[:, 0:1, :, :]
        mask5 = mask[:, 1:2, :, :]
        mask7 = mask[:, 2:3, :, :]

        # kernel_conf = self.kernel_conf_layer(feature)
        kernel_conf = self.softmax(feature)
        kernel_conf3 = kernel_conf[:, 0:1, :, :]
        kernel_conf5 = kernel_conf[:, 1:2, :, :]
        kernel_conf7 = kernel_conf[:, 2:3, :, :]

        conf = self.iter_conf_layer(feature)
        conf3 = conf[:, 0:4, :, :]
        conf5 = conf[:, 4:8, :, :]
        conf7 = conf[:, 8:12, :, :]
        conf3 = self.softmax(conf3)
        conf5 = self.softmax(conf5)
        conf7 = self.softmax(conf7)

        guide3 = self.iter_guide_layer3(feature)
        guide5 = self.iter_guide_layer5(feature)
        guide7 = self.iter_guide_layer7(feature)

        depth = feature
        depth3 = depth
        depth5 = depth
        depth7 = depth

        d3_list = [i for i in range(4)]
        d5_list = [i for i in range(4)]
        d7_list = [i for i in range(4)]

        guide3 = kernel_trans(guide3, self.encoder3)
        guide5 = kernel_trans(guide5, self.encoder5)
        guide7 = kernel_trans(guide7, self.encoder7)

        for i in range(12):
            depth3 = self.CSPN3(guide3, depth3, depth)
            depth3 = mask3*d + (1-mask3)*depth3
            depth5 = self.CSPN5(guide5, depth5, depth)
            depth5 = mask5*d + (1-mask5)*depth5
            depth7 = self.CSPN7(guide7, depth7, depth)
            depth7 = mask7*d + (1-mask7)*depth7

            if(i==2):
                d3_list[0] = depth3
                d5_list[0] = depth5
                d7_list[0] = depth7

            if(i==5):
                d3_list[1] = depth3
                d5_list[1] = depth5
                d7_list[1] = depth7

            if(i==8):
                d3_list[2] = depth3
                d5_list[2] = depth5
                d7_list[2] = depth7

            if(i==11):
                d3_list[3] = depth3
                d5_list[3] = depth5
                d7_list[3] = depth7

        refined_depth = \
        d3_list[0] * (kernel_conf3 * conf3[:, 0:1, :, :]) + \
        d3_list[1] * (kernel_conf3 * conf3[:, 1:2, :, :]) + \
        d3_list[2] * (kernel_conf3 * conf3[:, 2:3, :, :]) + \
        d3_list[3] * (kernel_conf3 * conf3[:, 3:4, :, :]) + \
        d5_list[0] * (kernel_conf5 * conf5[:, 0:1, :, :]) + \
        d5_list[1] * (kernel_conf5 * conf5[:, 1:2, :, :]) + \
        d5_list[2] * (kernel_conf5 * conf5[:, 2:3, :, :]) + \
        d5_list[3] * (kernel_conf5 * conf5[:, 3:4, :, :]) + \
        d7_list[0] * (kernel_conf7 * conf7[:, 0:1, :, :]) + \
        d7_list[1] * (kernel_conf7 * conf7[:, 1:2, :, :]) + \
        d7_list[2] * (kernel_conf7 * conf7[:, 2:3, :, :]) + \
        d7_list[3] * (kernel_conf7 * conf7[:, 3:4, :, :])

        refined_depth= self.cbam_conv(self.cbam(refined_depth))
        return   s_feature, b1_feature, b2_feature, b3_feature, b4_feature, b5_feature, b_feature, refined_depth

