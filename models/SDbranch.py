import torch
import torch.nn as nn
import torch.nn.init
import torch.optim.lr_scheduler
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
    @staticmethod
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

        self.apply(self.weight_init)

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
        dx1 = F.log_softmax(self.conv1_1_D(dx1))
        return dx1
