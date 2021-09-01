from torch import nn
import torch
import torch.nn.functional as F
from components.lossFunctions import *

class LambdaConv2(nn.Module):
    def __init__(self, in_channels, out_channels, heads=4, k=16, u=1, m=23):
        super(LambdaConv2, self).__init__()
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

        queries = self.queries(x).view(n_batch, self.heads, self.kk, w * h) # b, heads, k // heads, w * h
        softmax = self.softmax(self.keys(x).view(n_batch, self.kk, self.uu, w * h)) # b, k, uu, w * h
        values = self.values(x).view(n_batch, self.vv, self.uu, w * h) # b, v, uu, w * h
        # scale = 1 / torch.sqrt(w*h)
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

class LambdaConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, heads=4, k=8, u=1):
        super(LambdaConv, self).__init__()
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
        lambda_c = torch.einsum('bkum,bvum->bkv',( softmax, values)) # b, k, v
        y_c = torch.einsum('bhkn,bkv->bhvn', (queries, lambda_c)) # b, h, v, n

        # local context
        values = values.view(n_batch, self.uu*self.vv, w, h)
        lambda_p = F.conv2d(values, self.embedding, padding=(self.padding, self.padding))
        lambda_p = lambda_p.view(n_batch, self.kk, self.vv, w*h)
        y_p = torch.einsum('bhkn,bkvn->bhvn', (queries, lambda_p))

        out = y_c + y_p
        out = out.contiguous().view(n_batch, -1, w, h)

        if self.stride > 1:
            out = self.avgpool(out)

        return out

def Conv3X3(in_, out):
    return torch.nn.Conv2d(in_, out, 3, padding=1)


class ConvRelu(nn.Module):
    def __init__(self, in_, out):
        super().__init__()
        self.conv = Conv3X3(in_, out)
        self.activation = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x

class LambConvRelu(nn.Module):
    def __init__(self, in_, out):
        super().__init__()
        self.conv = LambdaConv(in_, out)
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

    def __init__(self, num_classes=1000):
        super(DeepCrack, self).__init__()

        self.down1 = Down(torch.nn.Sequential(
            ConvRelu(3,64),
            # ConvRelu(64,64),
            LambConvRelu(64,64),
        ))

        self.down2 = Down(torch.nn.Sequential(
            ConvRelu(64,128),
            # ConvRelu(128,128),
            LambConvRelu(128,128),
        ))

        self.down3 = Down(torch.nn.Sequential(
            ConvRelu(128,256),
            # ConvRelu(256,256),
            # ConvRelu(256,256),
            LambConvRelu(256, 256),
            LambConvRelu(256, 256),
        ))

        self.down4 = Down(torch.nn.Sequential(
            ConvRelu(256, 512),
            # ConvRelu(512, 512),
            # ConvRelu(512, 512),
            LambConvRelu(512, 512),
            LambConvRelu(512, 512),
        ))

        self.down5 = Down(torch.nn.Sequential(
            ConvRelu(512, 512),
            # ConvRelu(512, 512),
            # ConvRelu(512, 512),
            LambConvRelu(512, 512),
            LambConvRelu(512, 512),
        ))

        self.up1 = Up(torch.nn.Sequential(
            # ConvRelu(64, 64),
            LambConvRelu(64, 64),
            ConvRelu(64, 64),
        ))

        self.up2 = Up(torch.nn.Sequential(
            # ConvRelu(128, 128),
            LambConvRelu(128, 128),
            ConvRelu(128, 64),
        ))

        self.up3 = Up(torch.nn.Sequential(
            # ConvRelu(256, 256),
            # ConvRelu(256, 256),
            LambConvRelu(256, 256),
            LambConvRelu(256, 256),
            ConvRelu(256, 128),
        ))

        self.up4 = Up(torch.nn.Sequential(
            # ConvRelu(512, 512),
            # ConvRelu(512, 512),
            LambConvRelu(512, 512),
            LambConvRelu(512, 512),
            ConvRelu(512, 256),
        ))

        self.up5 = Up(torch.nn.Sequential(
            # ConvRelu(512, 512),
            # ConvRelu(512, 512),
            LambConvRelu(512, 512),
            LambConvRelu(512, 512),
            ConvRelu(512, 512),
        ))

        self.fuse5 = Fuse(ConvRelu(512 + 512, 64))
        self.fuse4 = Fuse(ConvRelu(512 + 256, 64))
        self.fuse3 = Fuse(ConvRelu(256 + 128, 64))
        self.fuse2 = Fuse(ConvRelu(128 + 64, 64))
        self.fuse1 = Fuse(ConvRelu(64 + 64, 64))

        self.final = Conv3X3(5,1)
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

        fuse5 = self.fuse5(down_inp=down5, up_inp=up5,size=size)
        fuse4 = self.fuse4(down_inp=down4, up_inp=up4,size=size)
        fuse3 = self.fuse3(down_inp=down3, up_inp=up3,size=size)
        fuse2 = self.fuse2(down_inp=down2, up_inp=up2,size=size)
        fuse1 = self.fuse1(down_inp=down1, up_inp=up1,size=size)


        output = self.final(torch.cat([fuse5,fuse4,fuse3,fuse2,fuse1],1))

        return fuse5, fuse4, fuse3, fuse2, fuse1, output

if __name__ == '__main__':
    inp = torch.randn((1,3,512,512))

    model = DeepCrack()

    out = model(inp)
