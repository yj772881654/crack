import torch
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from torch import nn, einsum
from einops import rearrange
from inspect import isfunction
# from mmcv.cnn import constant_init, kaiming_init
import torch.nn.init as init
# from utils import get_model_summary


# helpers
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

def last_zero_init(m):
    if isinstance(m, nn.Sequential):
        constant_init(m[-1], val=0)
        m[-1].inited = True
    else:
        constant_init(m, val=0)
        m.inited = True

class BaseConv(nn.Module):  # standalone self-attention
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, groups=1, bias=False):
        super(BaseConv, self).__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = (kernel_size-1)//2

        self.conv = nn.Conv2d(in_channels, self.out_channels, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, bias=bias)
        self.reset_parameters()

    def forward(self, x):
        out = self.conv(x)
        return out

    def reset_parameters(self):
        init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')


class LambdaConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=23, stride=1, heads=4, k=16, u=1):
        super(LambdaConv, self).__init__()
        self.kk, self.uu, self.vv, self.mm, self.heads = k, u, out_channels // heads, kernel_size, heads
        self.local_context = True if self.mm > 0 else False
        self.padding = (kernel_size - 1) // 2
        self.stride = stride

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
        if self.stride > 1:
            self.avgpool = nn.AvgPool2d(self.stride, self.stride)

        if self.local_context:
            self.embedding = nn.Parameter(torch.randn([self.kk, self.uu, 1, self.mm, self.mm]), requires_grad=True)
        else:
            self.embedding = nn.Parameter(torch.randn([self.kk, self.uu]), requires_grad=True)

    def forward(self, x):
        n_batch, C, w, h = x.size()

        queries = self.queries(x).view(n_batch, self.heads, self.kk, w * h) # b, heads, k // heads, w * h
        softmax = self.softmax(self.keys(x).view(n_batch, self.kk, self.uu, w * h)) # b, k, uu, w * h
        values = self.values(x).view(n_batch, self.vv, self.uu, w * h) # b, v, uu, w * h

        lambda_c = torch.einsum('bkum,bvum->bkv', softmax, values)
        y_c = torch.einsum('bhkn,bkv->bhvn', queries, lambda_c)

        if self.local_context:
            values = values.view(n_batch, self.uu, -1, w, h)
            lambda_p = F.conv3d(values, self.embedding, padding=(0, self.padding, self.padding))
            lambda_p = lambda_p.view(n_batch, self.kk, self.vv, w * h)
            y_p = torch.einsum('bhkn,bkvn->bhvn', queries, lambda_p)
        else:
            lambda_p = torch.einsum('ku,bvun->bkvn', self.embedding, values)
            y_p = torch.einsum('bhkn,bkvn->bhvn', queries, lambda_p)

        out = y_c + y_p
        out = out.contiguous().view(n_batch, -1, w, h)

        if self.stride > 1:
            out = self.avgpool(out)

        return out

class LambdaConv2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, heads=4, k=8, u=1):
        super(LambdaConv2, self).__init__()
        self.kk, self.uu, self.vv, self.mm, self.heads = k, u, out_channels // heads, kernel_size, heads
        self.local_context = True if kernel_size > 0 else False
        self.padding = (kernel_size - 1) // 2
        self.stride = stride

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
        if self.stride > 1:
            self.avgpool = nn.AvgPool2d(self.stride, self.stride)

        self.embedding = nn.Parameter(torch.randn([self.kk*self.vv, self.uu*self.vv, self.mm, self.mm]), requires_grad=True)

    def forward(self, x):
        n_batch, C, w, h = x.size()

        queries = self.queries(x).contiguous().view(n_batch, self.heads, self.kk, w * h) # b, heads, k, w * h
        softmax = self.softmax(self.keys(x).contiguous().view(n_batch, self.kk, self.uu, w * h)) # b, k, uu, w * h
        values = self.values(x).contiguous().view(n_batch, self.vv, self.uu, w * h) # b, v, uu, w * h

        # global context
        lambda_c = torch.einsum('bkum,bvum->bkv', softmax, values) # b, k, v
        y_c = torch.einsum('bhkn,bkv->bhvn', queries, lambda_c) # b, h, v, n

        # local context
        values = values.view(n_batch, self.uu*self.vv, w, h)
        lambda_p = F.conv2d(values, self.embedding, padding=(self.padding, self.padding))
        lambda_p = lambda_p.view(n_batch, self.kk, self.vv, w*h)
        y_p = torch.einsum('bhkn,bkvn->bhvn', queries, lambda_p)

        out = y_c + y_p
        out = out.contiguous().view(n_batch, -1, w, h)

        if self.stride > 1:
            out = self.avgpool(out)

        return out


# Test DeNonLocalAttention Code
tmp = torch.randn((2, 128, 32, 32))
# model = BaseConv(128, 64, 3, 1)
model = LambdaConv2(128, 128, 3, 1)
conv_out = model(tmp)

detail=get_model_summary(model, tmp)

print(conv_out.shape)
print(detail)