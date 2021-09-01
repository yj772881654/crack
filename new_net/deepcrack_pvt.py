import torch
import torch.nn as nn
import torch.nn.functional as F
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
    return torch.nn.Conv2d(in_, out , 3, padding=1)

class Mlp(nn.Module):
    def __init__(self, in_features, out_features, act_layer=nn.GELU, drop=0., linear=False):
        super().__init__()
        out_features = out_features or in_features
        # hidden_features = hidden_features or in_features
        # hidden_features = hidden_features or out_features
        hidden_features = out_features//4
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
        queries = self.queries(x).view(n_batch, self.heads, self.kk, w * h) # b, heads, k , w * h
        softmax = self.softmax(self.keys(x).view(n_batch, self.kk, self.uu, w * h)) # b, k, uu, w * h
        values = self.values(x).view(n_batch, self.vv, self.uu, w * h) # b, v, uu, w * h
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

    def __init__(self, in_chnnels, out_chnnels,  mlp_ratio=2., drop=0.,
                 drop_path=0., act_layer=nn.GELU,  linear=False):
        super(TFBlock, self).__init__()
        self.in_chnnels=in_chnnels
        self.out_chnnels=out_chnnels
        self.attn = LambdaAttention(
            in_channels=in_chnnels, out_channels=out_chnnels
        )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(out_chnnels * mlp_ratio)
        self.mlp = Mlp(in_features=in_chnnels, out_features=out_chnnels, act_layer=act_layer, drop=drop, linear=linear)
        self.apply(self._init_weights)

        self.shortcut = nn.Sequential()
        if self.in_chnnels != self.out_chnnels:
            self.shortcut = nn.Sequential(
                Conv1X1(self.in_chnnels,self.out_chnnels)
            )
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
        # print("in block {}".format(x.shape))
        x1=self.shortcut(x)
        x = x1 + self.drop_path(self.attn(x))
        x = x + self.drop_path(self.mlp(x))
        # print("out block {}".format(x.shape))
        return x


class Bottleneck(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.expansion = 4
        hidden_planes = in_planes // self.expansion
        self.conv1 = nn.Conv2d(in_planes, hidden_planes, kernel_size=1, bias=False)
        self.bn1 =nn.GroupNorm(hidden_planes//4, hidden_planes)  ##self.bn1 = nn.GroupNorm2d(hidden_planes//4, hidden_planes)

        self.conv2 = nn.ModuleList([TFBlock(hidden_planes, hidden_planes)])
        # if stride != 1 or in_planes != self.expansion * planes:
        #     self.conv2.append(nn.AvgPool2d(kernel_size=(3, 3), stride=stride, padding=(1, 1)))
        self.conv2.append(nn.GroupNorm(hidden_planes//4, hidden_planes)) ##self.bn2 = nn.GroupNorm2d(hidden_planes//4, hidden_planes)
        self.conv2.append(nn.GELU())   ##nn.GELU()
        self.conv2 = nn.Sequential(*self.conv2)
        self.conv3 = nn.Conv2d(hidden_planes, planes, kernel_size=1, bias=False)
        self.bn3 = nn.GroupNorm(planes//4, planes) ##self.bn3 = nn.GroupNorm2d(planes//4, planes)
        # self.shortcut = nn.Sequential()
        # if stride != 1 or in_planes != self.expansion*planes:
        #     self.shortcut = nn.Sequential(
        #         nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride),
        #         nn.BatchNorm2d(self.expansion*planes)
        #     )
    def forward(self, x):
        out = F.gelu(self.bn1(self.conv1(x)))  ##nn.GELU()
        out = self.conv2(out)
        out = F.gelu(self.bn3(self.conv3(out)))  ##nn.GELU()
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

class Down(nn.Module):

    def __init__(self, nn):
        super(Down,self).__init__()
        self.nn = nn
        self.maxpool_with_argmax = torch.nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

    def forward(self,inputs):
        down = self.nn(inputs)
        unpooled_shape = down.size()
        outputs, indices = self.maxpool_with_argmax(down)
        return outputs, down, indices, unpooled_shape

class Up(nn.Module):

    def __init__(self, nn):
        super().__init__()
        self.nn = nn
        self.unpool=torch.nn.MaxUnpool2d(2,2)

    def forward(self,inputs,indices,output_shape):
        outputs = self.unpool(inputs, indices=indices, output_size=output_shape)
        outputs = self.nn(outputs)
        return outputs

class Fuse(nn.Module):

    def __init__(self, nn):
        super().__init__()
        self.nn = nn
        # self.size = size
        self.conv = Conv3X3(64,1)

    def forward(self,down_inp,up_inp,size):
        outputs = torch.cat([down_inp, up_inp], 1)
        outputs = F.interpolate(outputs, size=size, mode='bilinear')
        outputs = self.nn(outputs)

        return self.conv(outputs)

class DeepCrack(nn.Module):

    def __init__(self):
        super(DeepCrack, self).__init__()

        self.down1 = Down(torch.nn.Sequential(
            ConvRelu(3, 64),
            PvtConvRelu(64,64),
        ))

        self.down2 = Down(torch.nn.Sequential(
            ConvRelu(64,128),
            PvtConvRelu(128,128),
        ))

        self.down3 = Down(torch.nn.Sequential(
            ConvRelu(128,256),
            PvtConvRelu(256,256),
            PvtConvRelu(256,256),
        ))

        self.down4 = Down(torch.nn.Sequential(
            ConvRelu(256, 512),
            PvtConvRelu(512, 512),
            PvtConvRelu(512, 512),
        ))

        self.down5 = Down(torch.nn.Sequential(
            PvtConvRelu(512, 512),
            PvtConvRelu(512, 512),
            PvtConvRelu(512, 512),
        ))

        self.up1 = Up(torch.nn.Sequential(
            PvtConvRelu(64, 64),
            PvtConvRelu(64, 64),
        ))

        self.up2 = Up(torch.nn.Sequential(
            PvtConvRelu(128, 128),
            ConvRelu(128, 64),
        ))

        self.up3 = Up(torch.nn.Sequential(
            PvtConvRelu(256, 256),
            PvtConvRelu(256, 256),
            ConvRelu(256, 128),
        ))

        self.up4 = Up(torch.nn.Sequential(
            PvtConvRelu(512, 512),
            PvtConvRelu(512, 512),
            ConvRelu(512, 256),
        ))

        self.up5 = Up(torch.nn.Sequential(
            PvtConvRelu(512, 512),
            PvtConvRelu(512, 512),
            PvtConvRelu(512, 512),
        ))

        self.fuse5 = Fuse(ConvRelu(512 + 512, 64))
        self.fuse4 = Fuse(ConvRelu(512 + 256, 64))
        self.fuse3 = Fuse(ConvRelu(256 + 128, 64))
        self.fuse2 = Fuse(ConvRelu(128 + 64, 64))
        self.fuse1 = Fuse(ConvRelu(64 + 64, 64))

        self.final = Conv3X3(5, 1)
    def calculate_loss(self, outputs, labels):
        loss = 0
        loss = cross_entropy_loss_RCF(outputs, labels)
        return loss
    def forward(self,inputs):
        size = [inputs.shape[2],inputs.shape[3]]
        # encoder part
        out, down1, indices_1, unpool_shape1 = self.down1(inputs)
        out, down2, indices_2, unpool_shape2 = self.down2(out)
        out, down3, indices_3, unpool_shape3 = self.down3(out)
        out, down4, indices_4, unpool_shape4 = self.down4(out)
        out, down5, indices_5, unpool_shape5 = self.down5(out)

        # decoder part
        up5 = self.up5(out, indices=indices_5, output_shape=unpool_shape5)
        up4 = self.up4(up5, indices=indices_4, output_shape=unpool_shape4)
        up3 = self.up3(up4, indices=indices_3, output_shape=unpool_shape3)
        up2 = self.up2(up3, indices=indices_2, output_shape=unpool_shape2)
        up1 = self.up1(up2, indices=indices_1, output_shape=unpool_shape1)

        fuse5 = self.fuse5(down_inp=down5,up_inp=up5,size=size)
        fuse4 = self.fuse4(down_inp=down4, up_inp=up4,size=size)
        fuse3 = self.fuse3(down_inp=down3, up_inp=up3,size=size)
        fuse2 = self.fuse2(down_inp=down2, up_inp=up2,size=size)
        fuse1 = self.fuse1(down_inp=down1, up_inp=up1,size=size)

        output = self.final(torch.cat([fuse5,fuse4,fuse3,fuse2,fuse1],1))

        return fuse5, fuse4, fuse3, fuse2, fuse1, output

if __name__ == '__main__':
    inp=torch.randn(1,3,512,512).cuda()
    model=DeepCrack().cuda()
    out=model(inp)
    print(model)