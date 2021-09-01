import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
import math
from components.lossFunctions import *


class DWConv(nn.Module):
    def __init__(self, dim=768,group_num=4):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim//group_num)

    def forward(self, x):
        x = self.dwconv(x)
        return x


def Conv1X1(in_, out):
    return torch.nn.Conv2d(in_, out, 1, padding=0)


def Conv3X3(in_, out):
    return torch.nn.Conv2d(in_, out, 3, padding=1)


class Mlp(nn.Module):
    def __init__(self, in_features, out_features, act_layer=nn.GELU, drop=0., linear=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = out_features // 4
        self.fc1 = Conv1X1(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = Conv1X1(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.linear = linear
        if self.linear:
            self.relu = nn.ReLU(inplace=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.fc1(x)
        if self.linear:
            x = self.relu(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class LambdaAttention(nn.Module):
    def __init__(self, in_channels, out_channels, heads=4, k=16, u=1, m=11):
        super(LambdaAttention, self).__init__()
        self.kk, self.uu, self.vv, self.mm, self.heads = k, u, out_channels // heads, m, heads
        self.local_context = True if m > 0 else False
        self.padding = (m - 1) // 2

        self.queries = nn.Sequential(
            nn.Conv2d(in_channels, k * heads, kernel_size=1, bias=False),
            nn.BatchNorm2d(k * heads)
        )
        self.keys = nn.Sequential(
            nn.Conv2d(in_channels, k * u, kernel_size=1, bias=False),
        )
        self.values = nn.Sequential(
            nn.Conv2d(in_channels, self.vv * u, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.vv * u)
        )

        self.softmax = nn.Softmax(dim=-1)

        if self.local_context:
            self.embedding = nn.Parameter(torch.randn([self.kk, self.uu, 1, m, m]), requires_grad=True)
        else:
            self.embedding = nn.Parameter(torch.randn([self.kk, self.uu]), requires_grad=True)

    def forward(self, x):
        n_batch, C, w, h = x.size()
        queries = self.queries(x).view(n_batch, self.heads, self.kk, w * h)  # b, heads, k , w * h
        softmax = self.softmax(self.keys(x).view(n_batch, self.kk, self.uu, w * h))  # b, k, uu, w * h
        values = self.values(x).view(n_batch, self.vv, self.uu, w * h)  # b, v, uu, w * h
        lambda_c = torch.einsum('bkum,bvum->bkv', (softmax, values))
        y_c = torch.einsum('bhkn,bkv->bhvn', (queries, lambda_c))
        if self.local_context:
            values = values.view(n_batch, self.uu, -1, w, h)
            lambda_p = F.conv3d(values, self.embedding, padding=(0, self.padding, self.padding))
            lambda_p = lambda_p.view(n_batch, self.kk, self.vv, w * h)
            y_p = torch.einsum('bhkn,bkvn->bhvn', (queries, lambda_p))
        else:
            lambda_p = torch.einsum('ku,bvun->bkvn', (self.embedding, values))
            y_p = torch.einsum('bhkn,bkvn->bhvn', (queries, lambda_p))

        out = y_c + y_p
        out = out.contiguous().view(n_batch, -1, w, h)

        return out


class TFBlock(nn.Module):

    def __init__(self, in_chnnels, out_chnnels, mlp_ratio=2., drop=0.,
                 drop_path=0., act_layer=nn.GELU, linear=False):
        super(TFBlock, self).__init__()
        self.in_chnnels = in_chnnels
        self.out_chnnels = out_chnnels
        self.attn = LambdaAttention(
            in_channels=in_chnnels, out_channels=out_chnnels
        )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(out_chnnels * mlp_ratio)
        self.mlp = Mlp(in_features=in_chnnels, out_features=out_chnnels, act_layer=act_layer, drop=drop, linear=linear)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = x + self.drop_path(self.attn(x))
        x = x + self.drop_path(self.mlp(x))
        return x


class Bottleneck(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.expansion = 4
        hidden_planes = in_planes // self.expansion
        self.conv1 = nn.Conv2d(in_planes, hidden_planes, kernel_size=1, bias=False)
        self.bn1 = nn.GroupNorm(hidden_planes // 4,
                                hidden_planes)  ##self.bn1 = nn.GroupNorm2d(hidden_planes//4, hidden_planes)
        self.conv2 = nn.ModuleList([TFBlock(hidden_planes, hidden_planes)])
        self.bn2 = nn.GroupNorm(hidden_planes // 4,
                                hidden_planes)  ##self.bn2 = nn.GroupNorm2d(hidden_planes//4, hidden_planes)
        self.conv2.append(nn.GELU())  ##nn.GELU()
        self.conv2 = nn.Sequential(*self.conv2)
        self.conv3 = nn.Conv2d(hidden_planes, planes, kernel_size=1, bias=False)
        self.bn3 = nn.GroupNorm(planes // 4, planes)  ##self.bn3 = nn.GroupNorm2d(planes//4, planes)
        self.GELU=nn.GELU()
    def forward(self, x):
        out = self.GELU(self.bn1(self.conv1(x)))  ##nn.GELU()
        out = self.conv2(out)
        out = self.GELU(self.bn3(self.conv3(out)))  ##nn.GELU()
        out += x
        return out


class PvtConvRelu(nn.Module):
    def __init__(self, in_, out):
        super().__init__()
        self.conv = Bottleneck(in_, out)
        self.activation = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x


class ConvRelu(nn.Module):
    def __init__(self, in_, out):
        super().__init__()
        self.conv = Conv3X3(in_, out)
        self.activation = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x


class LABlock(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(LABlock, self).__init__()
        self.W_1 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(output_channels)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(output_channels, output_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(output_channels),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs):
        sum = 0
        for input in inputs:
            sum += input
        sum = self.relu(sum)
        out = self.W_1(sum)
        psi = self.psi(out)  # Mask

        return psi


class Fuse(nn.Module):

    def __init__(self, nn, scale):
        super().__init__()
        self.nn = nn
        self.scale = scale
        self.conv = Conv3X3(64, 1)

    def forward(self, down_inp, up_inp, size, attention):
        outputs = torch.cat([down_inp, up_inp], 1)
        outputs = self.nn(outputs)
        outputs = attention * outputs
        outputs = self.conv(outputs)
        outputs = F.interpolate(outputs, scale_factor=self.scale, mode='bilinear')
        return outputs


class Down1(nn.Module):

    def __init__(self):
        super(Down1, self).__init__()
        self.nn1 = ConvRelu(3, 64)
        self.nn2 = PvtConvRelu(64, 64)
        self.maxpool_with_argmax = torch.nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

    def forward(self, inputs):
        scale1_1 = self.nn1(inputs)
        scale1_2 = self.nn2(scale1_1)
        unpooled_shape = scale1_2.size()
        outputs, indices = self.maxpool_with_argmax(scale1_2)
        return outputs, indices, unpooled_shape, scale1_1, scale1_2


class Down2(nn.Module):

    def __init__(self):
        super(Down2, self).__init__()
        self.nn1 = ConvRelu(64, 128)
        self.nn2 = PvtConvRelu(128, 128)
        self.maxpool_with_argmax = torch.nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

    def forward(self, inputs):
        scale2_1 = self.nn1(inputs)
        scale2_2 = self.nn2(scale2_1)
        unpooled_shape = scale2_2.size()
        outputs, indices = self.maxpool_with_argmax(scale2_2)
        return outputs, indices, unpooled_shape, scale2_1, scale2_2


class Down3(nn.Module):

    def __init__(self):
        super(Down3, self).__init__()

        self.nn1 = ConvRelu(128, 256)
        self.nn2 = PvtConvRelu(256, 256)
        self.nn3 = PvtConvRelu(256, 256)
        self.maxpool_with_argmax = torch.nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

    def forward(self, inputs):
        scale3_1 = self.nn1(inputs)
        scale3_2 = self.nn2(scale3_1)
        scale3_3 = self.nn2(scale3_2)
        unpooled_shape = scale3_3.size()
        outputs, indices = self.maxpool_with_argmax(scale3_3)
        return outputs, indices, unpooled_shape, scale3_1, scale3_2, scale3_3


class Down4(nn.Module):

    def __init__(self):
        super(Down4, self).__init__()

        self.nn1 = ConvRelu(256, 512)
        self.nn2 = PvtConvRelu(512, 512)
        self.nn3 = PvtConvRelu(512, 512)
        self.maxpool_with_argmax = torch.nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

    def forward(self, inputs):
        scale4_1 = self.nn1(inputs)
        scale4_2 = self.nn2(scale4_1)
        scale4_3 = self.nn2(scale4_2)
        unpooled_shape = scale4_3.size()
        outputs, indices = self.maxpool_with_argmax(scale4_3)
        return outputs, indices, unpooled_shape, scale4_1, scale4_2, scale4_3


class Down5(nn.Module):

    def __init__(self):
        super(Down5, self).__init__()

        self.nn1 = PvtConvRelu(512, 512)
        self.nn2 = PvtConvRelu(512, 512)
        self.nn3 = PvtConvRelu(512, 512)
        self.maxpool_with_argmax = torch.nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

    def forward(self, inputs):
        scale5_1 = self.nn1(inputs)
        scale5_2 = self.nn2(scale5_1)
        scale5_3 = self.nn2(scale5_2)
        unpooled_shape = scale5_3.size()
        outputs, indices = self.maxpool_with_argmax(scale5_3)
        return outputs, indices, unpooled_shape, scale5_1, scale5_2, scale5_3


class Up1(nn.Module):

    def __init__(self):
        super().__init__()
        self.nn1 = PvtConvRelu(64, 64)
        self.nn2 = PvtConvRelu(64, 64)
        self.unpool = torch.nn.MaxUnpool2d(2, 2)

    def forward(self, inputs, indices, output_shape):
        outputs = self.unpool(inputs, indices=indices, output_size=output_shape)
        scale1_3 = self.nn1(outputs)
        scale1_4 = self.nn2(scale1_3)
        return scale1_3, scale1_4


class Up2(nn.Module):

    def __init__(self):
        super().__init__()
        self.nn1 = PvtConvRelu(128, 128)
        self.nn2 = ConvRelu(128, 64)
        self.unpool = torch.nn.MaxUnpool2d(2, 2)

    def forward(self, inputs, indices, output_shape):
        outputs = self.unpool(inputs, indices=indices, output_size=output_shape)
        scale2_3 = self.nn1(outputs)
        scale2_4 = self.nn2(scale2_3)
        return scale2_3, scale2_4


class Up3(nn.Module):

    def __init__(self):
        super().__init__()
        self.nn1 = PvtConvRelu(256, 256)
        self.nn2 = PvtConvRelu(256, 256)
        self.nn3 = ConvRelu(256, 128)
        self.unpool = torch.nn.MaxUnpool2d(2, 2)

    def forward(self, inputs, indices, output_shape):
        outputs = self.unpool(inputs, indices=indices, output_size=output_shape)
        scale3_4 = self.nn1(outputs)
        scale3_5 = self.nn2(scale3_4)
        scale3_6 = self.nn3(scale3_5)
        return scale3_4, scale3_5, scale3_6


class Up4(nn.Module):

    def __init__(self):
        super().__init__()
        self.nn1 = PvtConvRelu(512, 512)
        self.nn2 = PvtConvRelu(512, 512)
        self.nn3 = ConvRelu(512, 256)
        self.unpool = torch.nn.MaxUnpool2d(2, 2)

    def forward(self, inputs, indices, output_shape):
        outputs = self.unpool(inputs, indices=indices, output_size=output_shape)
        scale4_4 = self.nn1(outputs)
        scale4_5 = self.nn2(scale4_4)
        scale4_6 = self.nn3(scale4_5)
        return scale4_4, scale4_5, scale4_6


class Up5(nn.Module):

    def __init__(self):
        super().__init__()
        self.nn1 = PvtConvRelu(512, 512)
        self.nn2 = PvtConvRelu(512, 512)
        self.nn3 = PvtConvRelu(512, 512)
        self.unpool = torch.nn.MaxUnpool2d(2, 2)

    def forward(self, inputs, indices, output_shape):
        outputs = self.unpool(inputs, indices=indices, output_size=output_shape)
        scale5_4 = self.nn1(outputs)
        scale5_5 = self.nn2(scale5_4)
        scale5_6 = self.nn3(scale5_5)
        return scale5_4, scale5_5, scale5_6


class DeepCrack(nn.Module):

    def __init__(self):
        super(DeepCrack, self).__init__()

        self.down1 = Down1()
        self.down2 = Down2()
        self.down3 = Down3()
        self.down4 = Down4()
        self.down5 = Down5()

        self.up1 = Up1()
        self.up2 = Up2()
        self.up3 = Up3()
        self.up4 = Up4()
        self.up5 = Up5()

        self.fuse5 = Fuse(ConvRelu(512 + 512, 64), scale=16)
        self.fuse4 = Fuse(ConvRelu(512 + 256, 64), scale=8)
        self.fuse3 = Fuse(ConvRelu(256 + 128, 64), scale=4)
        self.fuse2 = Fuse(ConvRelu(128 + 64, 64), scale=2)
        self.fuse1 = Fuse(ConvRelu(64 + 64, 64), scale=1)

        self.final = Conv1X1(5, 1)

        self.LABlock_1 = LABlock(64, 64)
        self.LABlock_2 = LABlock(128, 64)
        self.LABlock_3 = LABlock(256, 64)
        self.LABlock_4 = LABlock(512, 64)
        self.LABlock_5 = LABlock(512, 64)

    def calculate_loss(self, outputs, labels):
        loss = 0
        loss = cross_entropy_loss_RCF(outputs, labels)
        return loss

    def forward(self, inputs):
        # encoder part
        out, indices_1, unpool_shape1, scale1_1, scale1_2 = self.down1(inputs)
        out, indices_2, unpool_shape2, scale2_1, scale2_2 = self.down2(out)
        out, indices_3, unpool_shape3, scale3_1, scale3_2, scale3_3 = self.down3(out)
        out, indices_4, unpool_shape4, scale4_1, scale4_2, scale4_3 = self.down4(out)
        out, indices_5, unpool_shape5, scale5_1, scale5_2, scale5_3 = self.down5(out)
        # decoder part
        scale5_4, scale5_5, up5 = self.up5(out, indices=indices_5, output_shape=unpool_shape5)
        scale4_4, scale4_5, up4 = self.up4(up5, indices=indices_4, output_shape=unpool_shape4)
        scale3_4, scale3_5, up3 = self.up3(up4, indices=indices_3, output_shape=unpool_shape3)
        scale2_3, up2 = self.up2(up3, indices=indices_2, output_shape=unpool_shape2)
        scale1_3, up1 = self.up1(up2, indices=indices_1, output_shape=unpool_shape1)
        # attention part
        attention1 = self.LABlock_1([scale1_1, scale1_3])
        attention2 = self.LABlock_2([scale2_1, scale2_3])
        attention3 = self.LABlock_3([scale3_1, scale3_2, scale3_4, scale3_5])
        attention4 = self.LABlock_4([scale4_1, scale4_2, scale4_4, scale4_5])
        attention5 = self.LABlock_5([scale5_1, scale5_2, scale5_4, scale5_5])
        # fuse part
        fuse5 = self.fuse5(down_inp=scale5_3, up_inp=up5, size=[inputs.shape[2], inputs.shape[3]], attention=attention5)
        fuse4 = self.fuse4(down_inp=scale4_3, up_inp=up4, size=[inputs.shape[2], inputs.shape[3]], attention=attention4)
        fuse3 = self.fuse3(down_inp=scale3_3, up_inp=up3, size=[inputs.shape[2], inputs.shape[3]], attention=attention3)
        fuse2 = self.fuse2(down_inp=scale2_2, up_inp=up2, size=[inputs.shape[2], inputs.shape[3]], attention=attention2)
        fuse1 = self.fuse1(down_inp=scale1_2, up_inp=up1, size=[inputs.shape[2], inputs.shape[3]], attention=attention1)

        output = self.final(torch.cat([fuse5, fuse4, fuse3, fuse2, fuse1], 1))

        return fuse5, fuse4, fuse3, fuse2, fuse1, output


if __name__ == '__main__':
    inp = torch.randn(1, 3, 512, 512).cuda()
    model = DeepCrack().cuda()
    out=model(inp)
    # print(model)

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
    print('Trainable Parameters: %.3fM' % parameters)
