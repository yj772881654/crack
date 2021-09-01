import torch.nn.functional as F
from torch import nn
from components.lossFunctions import *

from components.Basic_blocks import *


class Conv_residual_conv(nn.Module):

    def __init__(self,in_dim,out_dim,act_fn):
        super(Conv_residual_conv,self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        act_fn = act_fn

        self.conv_1 = conv_block(self.in_dim,self.out_dim,act_fn)
        self.conv_2 = conv_block_3(self.out_dim,self.out_dim,act_fn)
        self.conv_3 = conv_block(self.out_dim,self.out_dim,act_fn)

    def forward(self,input):
        conv_1 = self.conv_1(input)
        conv_2 = self.conv_2(conv_1)
        res = conv_1 + conv_2
        conv_3 = self.conv_3(res)
        return conv_3


class FusionGenerator(nn.Module):

    def __init__(self,input_nc, output_nc, ngf):    # output_nc 表示最终的输出层数
        super(FusionGenerator,self).__init__()
        self.in_dim = input_nc
        self.out_dim = ngf
        self.final_out_dim = output_nc
        act_fn = nn.LeakyReLU(0.2, inplace=True)
        act_fn_2 = nn.ReLU()

        print("\n------Initiating FusionNet------\n")

        # encoder

        self.down_1 = Conv_residual_conv(self.in_dim, self.out_dim, act_fn)
        self.pool_1 = maxpool()
        self.down_2 = Conv_residual_conv(self.out_dim, self.out_dim * 2, act_fn)
        self.pool_2 = maxpool()
        self.down_3 = Conv_residual_conv(self.out_dim * 2, self.out_dim * 4, act_fn)
        self.pool_3 = maxpool()
        self.down_4 = Conv_residual_conv(self.out_dim * 4, self.out_dim * 8, act_fn)
        self.pool_4 = maxpool()

        # bridge

        self.bridge = Conv_residual_conv(self.out_dim * 8, self.out_dim * 16, act_fn)

        # decoder

        self.deconv_1 = conv_trans_block(self.out_dim * 16, self.out_dim * 8, act_fn_2)  # out_dim =16
        self.up_1 = Conv_residual_conv(self.out_dim * 8, self.out_dim * 8, act_fn_2)
        self.deconv_2 = conv_trans_block(self.out_dim * 8, self.out_dim * 4, act_fn_2)
        self.up_2 = Conv_residual_conv(self.out_dim * 4, self.out_dim * 4, act_fn_2)
        self.deconv_3 = conv_trans_block(self.out_dim * 4, self.out_dim * 2, act_fn_2)
        self.up_3 = Conv_residual_conv(self.out_dim * 2, self.out_dim * 2, act_fn_2)
        self.deconv_4 = conv_trans_block(self.out_dim * 2, self.out_dim, act_fn_2)
        self.up_4 = Conv_residual_conv(self.out_dim, self.out_dim, act_fn_2)

        # output

        self.out = nn.Conv2d(self.out_dim,self.final_out_dim, kernel_size=3, stride=1, padding=1)
        # self.out_2 = nn.Tanh()



        # decoder2
        self.dsn1 = nn.Conv2d(128, 1, 1)
        self.dsn2 = nn.Conv2d(64, 1, 1)
        self.dsn3 = nn.Conv2d(32, 1, 1)
        self.dsn4 = nn.Conv2d(16, 1, 1)
        # self.dsn5 = nn.Conv2d(512, 1, 1)



        # initialization

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
                m.bias.data.fill_(0)
            
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)


    def forward(self,input):

        h = input.size(2)
        w = input.size(3)

        down_1 = self.down_1(input)
        pool_1 = self.pool_1(down_1)
        down_2 = self.down_2(pool_1)
        pool_2 = self.pool_2(down_2)
        down_3 = self.down_3(pool_2)
        pool_3 = self.pool_3(down_3)
        down_4 = self.down_4(pool_3)
        pool_4 = self.pool_4(down_4)

        bridge = self.bridge(pool_4)

        deconv_1 = self.deconv_1(bridge)
        skip_1 = (deconv_1 + down_4)/2   # 1/8 C=128

        # 求第一个side 的 loss  1.降维 128->1  2.使用双线性插值进行上采样  2.sigmoid得出概率
        single_skip_1 = self.dsn1(skip_1)
        side1 = F.upsample_bilinear(single_skip_1, size=(h, w))
        side1 = F.sigmoid(side1)

        up_1 = self.up_1(skip_1)
        deconv_2 = self.deconv_2(up_1)
        skip_2 = (deconv_2 + down_3)/2   # 1/4 C=64

        # 求第二个side 的 loss  1.降维 64->1  2.使用双线性插值进行上采样  2.sigmoid得出概率
        single_skip_2 = self.dsn2(skip_2)
        side2 = F.upsample_bilinear(single_skip_2, size=(h, w))
        side2 = F.sigmoid(side2)


        up_2 = self.up_2(skip_2)
        deconv_3 = self.deconv_3(up_2)
        skip_3 = (deconv_3 + down_2)/2   # 1/2 C=32

        # 求第三个side 的 loss  1.降维 32->1  2.使用双线性插值进行上采样  2.sigmoid得出概率
        single_skip_3 = self.dsn3(skip_3)
        side3 = F.upsample_bilinear(single_skip_3, size=(h, w))
        side3 = F.sigmoid(side3)


        up_3 = self.up_3(skip_3)
        deconv_4 = self.deconv_4(up_3)
        skip_4 = (deconv_4 + down_1)/2   # 1 C=16

        # 求第四个side 的 loss  1.降维 16->1  2.使用双线性插值进行上采样  2.sigmoid得出概率
        single_skip_4 = self.dsn4(skip_4)
        side4 = F.upsample_bilinear(single_skip_4, size=(h, w))
        side4 = F.sigmoid(side4)



        up_4 = self.up_4(skip_4)

        out = self.out(up_4)
        finalout = F.sigmoid(out)

        #out = torch.clamp(out, min=-1, max=1)

        return side1 , side2 , side3 , side4 ,finalout
    
    def calculate_loss(self,outputs,labels):
        side_output1 = outputs[0]
        side_output2 = outputs[1]
        side_output3 = outputs[2]
        side_output4 = outputs[3]
        final_output = outputs[4]

        loss_side1 = bce2d(side_output1, labels)
        loss_side2 = bce2d(side_output2, labels)
        loss_side3 = bce2d(side_output3, labels)
        loss_side4 = bce2d(side_output4, labels)
        final_loss = bce2d(final_output, labels)

        loss = (loss_side1 + 2 * loss_side2 + 4 * loss_side3 + 8 * loss_side4 + final_loss)

        return loss