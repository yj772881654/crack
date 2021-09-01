import torch.nn.functional as F
from torch import nn
from components.lossFunctions import *

from components.Basic_blocks import *

from components.resnext import *


# class Attention3to1(nn.Module):
#     def __init__(self, F_one, F_two, F_int):
#         super(Attention_block, self).__init__()
#         self.W_g = nn.Sequential(
#             nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
#             nn.BatchNorm2d(F_int)
#         )
#
#         self.W_x = nn.Sequential(
#             nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
#             nn.BatchNorm2d(F_int)
#         )
#
#         self.psi = nn.Sequential(
#             nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
#             nn.BatchNorm2d(1),
#             nn.Sigmoid()
#         )
#
#         self.relu = nn.ReLU(inplace=True)
#
#     def forward(self, g, x):
#         g1 = self.W_g(g)
#         x1 = self.W_x(x)
#         psi = self.relu(g1 + x1)
#         psi = self.psi(psi)
#
#         return x * psi



class MSBlock(nn.Module):
        def __init__(self, c_in, rate=4):
            super(MSBlock, self).__init__()
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

class ResFusionGenerator(nn.Module):
        def __init__(self, rate=4):
            super(ResFusionGenerator, self).__init__()

            t = 1

            # self.features = vgg16_c.VGG16_C(pretrain, logger)
            self.resnext = resnext50()
            self.msblock1 = MSBlock(256, rate)

            self.conv1_1_down = nn.Conv2d(32 * t, 1, (1, 1), stride=1)
            self.conv1_2_down = nn.Conv2d(32 * t, 1, (1, 1), stride=1)

            self.msblock2 = MSBlock(512, rate)

            self.conv2_1_down = nn.Conv2d(32 * t, 1, (1, 1), stride=1)
            self.conv2_2_down = nn.Conv2d(32 * t, 1, (1, 1), stride=1)


            self.msblock3 = MSBlock(1024, rate)


            self.conv3_1_down = nn.Conv2d(32 * t, 1, (1, 1), stride=1)
            self.conv3_2_down = nn.Conv2d(32 * t, 1, (1, 1), stride=1)

            self.msblock4 = MSBlock(2048, rate)

            self.conv4_1_down = nn.Conv2d(32 * t, 1, (1, 1), stride=1)
            self.conv4_2_down = nn.Conv2d(32 * t, 1, (1, 1), stride=1)

            self.upsample_4 = nn.Upsample(scale_factor=4, mode='bilinear')  # ( k - stride) = 4
            self.upsample_8 = nn.Upsample(scale_factor=8, mode='bilinear')
            self.upsample_16 = nn.Upsample(scale_factor=16, mode='bilinear')
            self.upsample_32 = nn.Upsample(scale_factor=32, mode='bilinear')

            self.fuse = nn.Conv2d(8, 1, 1, stride=1)

            self.sigmoid = nn.Sigmoid()

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
            # print("x.shape-before =={}".format(x.data.shape))
            features = self.resnext(x)
            s1_sem = self.msblock1(features[0])
            s1 = self.conv1_1_down(s1_sem)
            s11 = self.conv1_2_down(s1_sem)
            s1 = nn.functional.interpolate(s1,size =(h,w),mode="bilinear")
            s11 = nn.functional.interpolate(s11,size=(h,w),mode="bilinear")

            # s1 = self.upsample_4(s1)
            # s11 = self.upsample_4(s11)

            # print("s1.shape-before =={}".format(s1.data.shape))

            s2_sem = self.msblock2(features[1])
            s2 = self.conv2_1_down(s2_sem)
            s21 = self.conv2_2_down(s2_sem)
            s2 = nn.functional.interpolate(s2, size=(h, w), mode="bilinear")
            s21 = nn.functional.interpolate(s21, size=(h, w), mode="bilinear")

            # s2 = self.upsample_8(s2)
            # s21 = self.upsample_8(s21)

            # print("s2.shape-before =={}".format(s2.data.shape))

            s3_sem = self.msblock3(features[2])
            s3 = self.conv3_1_down(s3_sem)
            s31 = self.conv3_2_down(s3_sem)

            s3 = nn.functional.interpolate(s3, size=(h, w), mode="bilinear")
            s31 = nn.functional.interpolate(s31, size=(h, w), mode="bilinear")
            # s3 = self.upsample_16(s3)
            # s31 = self.upsample_16(s31)

            # print("s3.shape-before =={}".format(s3.data.shape))

            s4_sem = self.msblock4(features[3])
            s4 = self.conv4_1_down(s4_sem)
            s41 = self.conv4_2_down(s4_sem)

            s4 = nn.functional.interpolate(s4, size=(h, w), mode="bilinear")
            s41 = nn.functional.interpolate(s41, size=(h, w), mode="bilinear")

            # s4 = self.upsample_32(s4)
            # s41 = self.upsample_32(s41)

            attention1 = sigmoid(s2 + s3 + s4)
            attention2 = sigmoid(s1 + s3 + s4)
            attention3 = sigmoid(s1 + s2 + s4)
            attention4 = sigmoid(s1 + s2 + s3)
            # attention5 = sigmoid(s1 + s2 + s3 + s4)

            s1 = attention1 * s1
            s2 = attention2 * s2
            s3 = attention3 * s3
            s4 = attention4 * s4


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
            # fuse = attention5 * fuse

            return [p1_1, p2_1, p3_1, p4_1, p1_2, p2_2, p3_2, p4_2, fuse,attention1,attention2,attention3,attention4]


        def calculate_loss(self, outputs, labels):
            loss = 0
            for k in range(8):
                loss += 0.5 * cross_entropy_loss_RCF(outputs[k], labels)
            loss += 1.1 * cross_entropy_loss_RCF(outputs[8], labels)

            # loss = (loss_side1 + 2 * loss_side2 + 4 * loss_side3 + 8 * loss_side4 + final_loss)

            return loss

if __name__ == '__main__':
    model = ResFusionGenerator()
    a=torch.rand(2,3,384,544)
    a=torch.autograd.Variable(a)

    for x in model(a):
        print(x.data.shape)