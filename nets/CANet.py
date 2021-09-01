import torch.nn.functional as F
from torch import nn
from components.lossFunctions import *

from components.Basic_blocks import *

from components.resnext_16out import *

from components.Attentions import *


class MVEBlock(nn.Module):  # 扩张卷积模块

        def __init__(self, c_in, rate=4):
            super(MVEBlock, self).__init__()
            c_out = c_in
            self.rate = rate

            self.conv = nn.Conv2d(c_in, 32, 3, stride=1, padding=1)
            self.relu = nn.ReLU(inplace=True)

            dilation = self.rate * 1 if self.rate >= 1 else 1  # 4   rate*1
            self.conv1 = nn.Conv2d(32, 32, 3, stride=1, dilation=dilation, padding=dilation)
            self.relu1 = nn.ReLU(inplace=True)

            dilation = self.rate * 2 if self.rate >= 1 else 1  # 8   rate*2
            self.conv2 = nn.Conv2d(32, 32, 3, stride=1, dilation=dilation, padding=dilation)
            self.relu2 = nn.ReLU(inplace=True)

            dilation = self.rate * 3 if self.rate >= 1 else 1  # 12  rate*3
            self.conv3 = nn.Conv2d(32, 32, 3, stride=1, dilation=dilation, padding=dilation)
            self.relu3 = nn.ReLU(inplace=True)

            self._initialize_weights()

        def forward(self, x):
            o = self.relu(self.conv(x))
            o1 = self.relu1(self.conv1(o))
            o2 = self.relu2(self.conv2(o))
            o3 = self.relu3(self.conv3(o))
            out = o + o1 + o2 + o3
            return out

        def _initialize_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    m.weight.data.normal_(0, 0.01)
                    if m.bias is not None:
                        m.bias.data.zero_()

class CANetGenerator(nn.Module):
        def __init__(self, backbone='resnext50', rate=4):
            super(CANetGenerator, self).__init__()

            t = 1

            # self.features = vgg16_c.VGG16_C(pretrain, logger)
            if backbone == 'resnext50':
               self.resnext = resnext50()
            elif backbone == 'resnext101':
               self.resnext = resnext101()

            self.LABlock_1 = LABlock(256)
            self.LABlock_2 = LABlock(512)
            self.LABlock_3 = LABlock(1024)
            self.LABlock_4 = LABlock(2048) # LA 输入通道数

            self.SABlock1_1 = SABlock(3)
            self.SABlock2_1 = SABlock(3)
            self.SABlock3_1 = SABlock(3)
            self.SABlock4_1 = SABlock(3)

            self.SABlock1_2 = SABlock(3)
            self.SABlock2_2 = SABlock(3)
            self.SABlock3_2 = SABlock(3)
            self.SABlock4_2 = SABlock(3)

            self.msblock1 = MVEBlock(256, rate)
            self.conv1_1_down = nn.Conv2d(32 * t, 1, (1, 1), stride=1)
            self.conv1_2_down = nn.Conv2d(32 * t, 1, (1, 1), stride=1)

            self.msblock2 = MVEBlock(512, rate)
            self.conv2_1_down = nn.Conv2d(32 * t, 1, (1, 1), stride=1)
            self.conv2_2_down = nn.Conv2d(32 * t, 1, (1, 1), stride=1)

            self.msblock3 = MVEBlock(1024, rate)
            self.conv3_1_down = nn.Conv2d(32 * t, 1, (1, 1), stride=1)
            self.conv3_2_down = nn.Conv2d(32 * t, 1, (1, 1), stride=1)

            self.msblock4 = MVEBlock(2048, rate)
            self.conv4_1_down = nn.Conv2d(32 * t, 1, (1, 1), stride=1)
            self.conv4_2_down = nn.Conv2d(32 * t, 1, (1, 1), stride=1)

            self.upsample_4 = nn.Upsample(scale_factor=4, mode='bilinear')
            self.upsample_8 = nn.Upsample(scale_factor=8, mode='bilinear')
            self.upsample_16 = nn.Upsample(scale_factor=16, mode='bilinear')
            self.upsample_32 = nn.Upsample(scale_factor=32, mode='bilinear')

            self.fuse = nn.Conv2d(8, 1, 1, stride=1)

            self.sigmoid = nn.Sigmoid()
            self.relu = nn.ReLU(inplace=True)

            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    m.weight.data.normal_(0.0, 0.02)
                    if m.bias is not None:
                        m.bias.data.zero_()

                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.normal_(1.0, 0.02)
                    if m.bias is not None:
                        m.bias.data.zero_()

        def forward(self, x):
            _,_,h,w = x.data.shape

            features = self.resnext(x) #backbone获取特征

            # print(features.size())
            # Scale 1
            s1_sem = self.msblock1(features[2])
            # LayerAttention
            s1_sem = self.LABlock_1([features[0], features[1]], s1_sem)

            s1 = self.conv1_1_down(s1_sem)
            s11 = self.conv1_2_down(s1_sem)

            s1 = nn.functional.interpolate(s1,size =(h,w),mode="bilinear")
            s11 = nn.functional.interpolate(s11,size=(h,w),mode="bilinear")

            # Scale 2
            s2_sem = self.msblock2(features[6])
            # LayerAttention
            s2_sem = self.LABlock_2([features[3], features[4], features[5]], s2_sem)

            s2 = self.conv2_1_down(s2_sem)
            s21 = self.conv2_2_down(s2_sem)

            s2 = nn.functional.interpolate(s2, size=(h, w), mode="bilinear")
            s21 = nn.functional.interpolate(s21, size=(h, w), mode="bilinear")

            # Scale 3
            s3_sem = self.msblock3(features[12])
            # LayerAttention
            s3_sem = self.LABlock_3([features[7], features[8], features[9], features[10], features[11]], s3_sem)

            s3 = self.conv3_1_down(s3_sem)
            s31 = self.conv3_2_down(s3_sem)

            s3 = nn.functional.interpolate(s3, size=(h, w), mode="bilinear")
            s31 = nn.functional.interpolate(s31, size=(h, w), mode="bilinear")

            # Scale 4
            s4_sem = self.msblock4(features[15])
            # LayerAttention
            s4_sem = self.LABlock_4([features[13], features[14]], s4_sem)

            s4 = self.conv4_1_down(s4_sem)
            s41 = self.conv4_2_down(s4_sem)

            s4 = nn.functional.interpolate(s4, size=(h, w), mode="bilinear")
            s41 = nn.functional.interpolate(s41, size=(h, w), mode="bilinear")

            #Scale Attention 模块
            s1 = self.SABlock1_1([s2, s3, s4], s1)
            s2 = self.SABlock2_1([s1, s3, s4], s2)
            s3 = self.SABlock3_1([s1, s2, s4], s3)
            s4 = self.SABlock4_1([s1, s2, s3], s4)

            s11 = self.SABlock1_2([s21, s31, s41], s11)
            s21 = self.SABlock2_2([s11, s31, s41], s21)
            s31 = self.SABlock3_2([s11, s21, s41], s31)
            s41 = self.SABlock4_2([s11, s21, s31], s41)

            o1, o2, o3, o4 = s1.detach(), s2.detach(), s3.detach(), s4.detach()
            o11, o21, o31, o41 = s11.detach(), s21.detach(), s31.detach(), s41.detach()

            p1_1 = s1
            p2_1 = s2 + o1
            p3_1 = s3 + o2 + o1
            p4_1 = s4 + o3 + o2 + o1
            p1_2 = s11 + o21 + o31 + o41
            p2_2 = s21 + o31 + o41
            p3_2 = s31 + o41
            p4_2 = s41

            fuse = self.fuse(torch.cat([p1_1, p2_1, p3_1, p4_1, p1_2, p2_2, p3_2, p4_2], 1))

            return [p1_1, p2_1, p3_1, p4_1, p1_2, p2_2, p3_2, p4_2, fuse]


        def calculate_loss(self, outputs, labels):
            loss = 0
            for k in range(8):
                loss += 0.5 * cross_entropy_loss_RCF(outputs[k], labels)
            loss += 1.1 * cross_entropy_loss_RCF(outputs[-1], labels)
            return loss

if __name__ == '__main__':
    model = CANetGenerator()
    a=torch.rand(2,3,384,544)
    a=torch.autograd.Variable(a)

    # print(model(a).size)
    for x in model(a):
        print(x.data.shape)