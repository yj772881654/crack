from torch import nn
import torch
import torch.nn.functional as F


def Conv3X3(in_, out):
    return torch.nn.Conv2d(in_, out, 3, padding=1)
def Conv1X1(in_, out):
    return torch.nn.Conv2d(in_, out, 1, padding=0)
def crop(data1, data2, crop_h, crop_w):
    _, _, h1, w1 = data1.size()
    print(h1)
    print(w1)
    h2, w2 = data2
    print(h2)
    print(w2)
    assert(h2 <= h1 and w2 <= w1)
    data = data1[:, :, crop_h:crop_h+h2, crop_w:crop_w+w2]
    return data

class LambdaConv(nn.Module):
    def __init__(self, in_channels, out_channels, heads=4, k=16, u=1, m=23):
        super(LambdaConv, self).__init__()
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

class LambdaBottleneck(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(LambdaBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.ModuleList([LambdaConv(planes, planes)])
        if stride != 1 or in_planes != self.expansion * planes:
            self.conv2.append(nn.AvgPool2d(kernel_size=(3, 3), stride=stride, padding=(1, 1)))
        self.conv2.append(nn.BatchNorm2d(planes))
        self.conv2.append(nn.ReLU())
        self.conv2 = nn.Sequential(*self.conv2)

        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.conv2(out)
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

def ConvLambda(in_, out):
    return LambdaBottleneck(in_,out)



class LABlock(nn.Module):
    def __init__(self, input_channels,output_channels):
        super(LABlock, self).__init__()
        self.W_1 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(output_channels)
        )

        # self.W_2 = nn.Sequential(
        #     nn.Conv2d(input_channels//2, 32, kernel_size=1, stride=1, padding=0, bias=True),
        #     nn.BatchNorm2d(32)
        # )

        self.psi = nn.Sequential(
            nn.Conv2d(output_channels, output_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(output_channels),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self,inputs):

        sum = 0
        for input in inputs:
            sum += input
        sum = self.relu(sum)
        out = self.W_1(sum)
        # out = self.W_2(out)

        psi = self.psi(out) # Mask

        return  psi

# class ConvRelu(nn.Module):
    # def __init__(self, in_, out):results
    #     super().__init__()
    #     # self.conv = Conv3X3(in_, out)
    #     # self.lambdaCov = LambdaConv(in_, out)
    #     self.lambdaCov = ConvLambda(in_, out)
    #     self.activation = torch.nn.ReLU(inplace=True)
    #
    # def forward(self, x):
    #     x = self.lambdaCov(x)
    #     # x = self.conv(x)
    #     x = self.activation(x)
    #     return x


# class Down(nn.Module):
#     def __init__(self,input_channels,output_channels,blocks_num):
#         super(Down,self).__init__()
#         self.nn1 = ConvRelu(input_channels, output_channels)
#         self.nn2 = ConvRelu(output_channels, output_channels)
#         self.blocks_num = blocks_num
#         if self.blocks_num == 3:
#             self.nn3 = ConvRelu(output_channels, output_channels)
#         self.maxpool_with_argmax = torch.nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
#
#     def forward(self,inputs):
#         scale1 = self.nn1(inputs)
#         scale2 = self.nn2(scale1)
#         if self.blocks_num == 3:
#
#         unpooled_shape = scale2.size()
#         outputs, indices = self.maxpool_with_argmax(scale2)
#         return outputs, indices, unpooled_shape,scale1,scale1



class Down1(nn.Module):

    def __init__(self):
        super(Down1,self).__init__()
        self.nn1 = ConvRelu(3, 64)
        self.nn2 = ConvRelu(64, 64)
        self.maxpool_with_argmax = torch.nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

    def forward(self,inputs):
        scale1_1 = self.nn1(inputs)
        scale1_2 = self.nn2(scale1_1)
        unpooled_shape = scale1_2.size()
        outputs, indices = self.maxpool_with_argmax(scale1_2)
        return outputs, indices, unpooled_shape,scale1_1,scale1_2

class Down2(nn.Module):

    def __init__(self):
        super(Down2,self).__init__()
        self.nn1 = ConvRelu(64, 128)
        self.nn2 = ConvRelu(128, 128)
        self.maxpool_with_argmax = torch.nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

    def forward(self,inputs):
        scale2_1 = self.nn1(inputs)
        scale2_2 = self.nn2(scale2_1)
        unpooled_shape = scale2_2.size()
        outputs, indices = self.maxpool_with_argmax(scale2_2)
        return outputs, indices, unpooled_shape, scale2_1, scale2_2

class Down3(nn.Module):

    def __init__(self):
        super(Down3,self).__init__()

        self.nn1 = ConvRelu(128, 256)
        self.nn2 = ConvRelu(256, 256)
        self.nn3 = ConvRelu(256, 256)
        self.maxpool_with_argmax = torch.nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

    def forward(self,inputs):
        scale3_1 = self.nn1(inputs)
        scale3_2 = self.nn2(scale3_1)
        scale3_3 = self.nn2(scale3_2)
        unpooled_shape = scale3_3.size()
        outputs, indices = self.maxpool_with_argmax(scale3_3)
        return outputs, indices, unpooled_shape, scale3_1, scale3_2, scale3_3

class Down4(nn.Module):

    def __init__(self):
        super(Down4,self).__init__()

        self.nn1 = ConvRelu(256, 512)
        self.nn2 = ConvRelu(512, 512)
        self.nn3 = ConvRelu(512, 512)
        self.maxpool_with_argmax = torch.nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

    def forward(self,inputs):
        scale4_1 = self.nn1(inputs)
        scale4_2 = self.nn2(scale4_1)
        scale4_3 = self.nn2(scale4_2)
        unpooled_shape = scale4_3.size()
        outputs, indices = self.maxpool_with_argmax(scale4_3)
        return outputs, indices, unpooled_shape, scale4_1, scale4_2, scale4_3

class Down5(nn.Module):

    def __init__(self):
        super(Down5,self).__init__()

        self.nn1 = ConvRelu(512, 512)
        self.nn2 = ConvRelu(512, 512)
        self.nn3 = ConvRelu(512, 512)
        self.maxpool_with_argmax = torch.nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

    def forward(self,inputs):
        scale5_1 = self.nn1(inputs)
        scale5_2 = self.nn2(scale5_1)
        scale5_3 = self.nn2(scale5_2)
        unpooled_shape = scale5_3.size()
        outputs, indices = self.maxpool_with_argmax(scale5_3)
        return outputs, indices, unpooled_shape, scale5_1, scale5_2, scale5_3

class Up1(nn.Module):

    def __init__(self):
        super().__init__()
        self.nn1 = ConvRelu(64, 64)
        self.nn2 = ConvRelu(64, 64)
        self.unpool=torch.nn.MaxUnpool2d(2,2)

    def forward(self,inputs,indices,output_shape):
        outputs = self.unpool(inputs, indices=indices, output_size=output_shape)
        scale1_3 = self.nn1(outputs)
        scale1_4 = self.nn2(scale1_3)
        return scale1_3, scale1_4

class Up2(nn.Module):

    def __init__(self):
        super().__init__()
        self.nn1 = ConvRelu(128, 128)
        self.nn2 = ConvRelu(128, 64)
        self.unpool=torch.nn.MaxUnpool2d(2,2)

    def forward(self,inputs,indices,output_shape):
        outputs = self.unpool(inputs, indices=indices, output_size=output_shape)
        scale2_3 = self.nn1(outputs)
        scale2_4 = self.nn2(scale2_3)
        return scale2_3, scale2_4

class Up3(nn.Module):

    def __init__(self):
        super().__init__()
        self.nn1 = ConvRelu(256, 256)
        self.nn2 = ConvRelu(256, 256)
        self.nn3 = ConvRelu(256, 128)
        self.unpool=torch.nn.MaxUnpool2d(2,2)

    def forward(self,inputs,indices,output_shape):
        outputs = self.unpool(inputs, indices=indices, output_size=output_shape)
        scale3_4 = self.nn1(outputs)
        scale3_5 = self.nn2(scale3_4)
        scale3_6 = self.nn3(scale3_5)
        return scale3_4, scale3_5, scale3_6

class Up4(nn.Module):

    def __init__(self):
        super().__init__()
        self.nn1 = ConvRelu(512, 512)
        self.nn2 = ConvRelu(512, 512)
        self.nn3 = ConvRelu(512, 256)
        self.unpool=torch.nn.MaxUnpool2d(2,2)

    def forward(self,inputs,indices,output_shape):
        outputs = self.unpool(inputs, indices=indices, output_size=output_shape)
        scale4_4 = self.nn1(outputs)
        scale4_5 = self.nn2(scale4_4)
        scale4_6 = self.nn3(scale4_5)
        return scale4_4, scale4_5, scale4_6

class Up5(nn.Module):

    def __init__(self):
        super().__init__()
        self.nn1 = ConvRelu(512, 512)
        self.nn2 = ConvRelu(512, 512)
        self.nn3 = ConvRelu(512, 512)
        self.unpool=torch.nn.MaxUnpool2d(2,2)

    def forward(self,inputs,indices,output_shape):
        outputs = self.unpool(inputs, indices=indices, output_size=output_shape)
        scale5_4 = self.nn1(outputs)
        scale5_5 = self.nn2(scale5_4)
        scale5_6 = self.nn3(scale5_5)
        return scale5_4, scale5_5, scale5_6

class Fuse(nn.Module):

    def __init__(self, nn, scale):
        super().__init__()
        self.nn = nn
        self.scale = scale

        #替换为 Bilinear
        # self.upsample = torch.nn.ConvTranspose2d(1, 1, 3, stride=self.scale)
        # self.conv = Conv3X3(64, 1)
        # 替换为 1X1
        self.conv = Conv1X1(64, 1)

    def forward(self,down_inp,up_inp,size, attention):
        outputs = torch.cat([down_inp, up_inp], 1)
        outputs = self.nn(outputs)
        outputs = attention * outputs
        outputs = self.conv(outputs)
        outputs = F.interpolate(outputs, scale_factor=self.scale, mode='bilinear')
        # outputs = self.upsample(outputs)
        # crop_outputs = crop(outputs,size,0,0)

        return outputs

class DeepCrack(nn.Module):

    def __init__(self, num_classes=1000):
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

        self.fuse5 = Fuse(Conv1X1(512 + 512, 64), scale=16)
        self.fuse4 = Fuse(Conv1X1(512 + 256, 64), scale=8)
        self.fuse3 = Fuse(Conv1X1(256 + 128, 64), scale=4)
        self.fuse2 = Fuse(Conv1X1(128 + 64, 64), scale=2)
        self.fuse1 = Fuse(Conv1X1(64 + 64, 64), scale=1)

        self.final = Conv1X1(5, 1)

        self.LABlock_1 = LABlock(64, 64)
        self.LABlock_2 = LABlock(128, 64)
        self.LABlock_3 = LABlock(256, 64)
        self.LABlock_4 = LABlock(512, 64)
        self.LABlock_5 = LABlock(512, 64)


    def forward(self,inputs):

        # print(inputs.shape)
        # encoder part

        # outputs, indices, unpooled_shape, scale1_1, scale1_2
        out, indices_1, unpool_shape1, scale1_1, scale1_2 = self.down1(inputs)
        out, indices_2, unpool_shape2, scale2_1, scale2_2 = self.down2(out)
        out, indices_3, unpool_shape3, scale3_1, scale3_2, scale3_3 = self.down3(out)
        out, indices_4, unpool_shape4, scale4_1, scale4_2, scale4_3 = self.down4(out)
        out, indices_5, unpool_shape5, scale5_1, scale5_2, scale5_3 = self.down5(out)
        # out, down2, indices_2, unpool_shape2 = self.down2(out)
        # out, down3, indices_3, unpool_shape3 = self.down3(out)
        # out, down4, indices_4, unpool_shape4 = self.down4(out)
        # out, down5, indices_5, unpool_shape5 = self.down5(out)

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

        # print(scale5_3.shape)
        # print(up5.shape)
        fuse5 = self.fuse5(down_inp=scale5_3, up_inp=up5,size=[inputs.shape[2],inputs.shape[3]], attention=attention5)
        fuse4 = self.fuse4(down_inp=scale4_3, up_inp=up4,size=[inputs.shape[2],inputs.shape[3]], attention=attention4)
        fuse3 = self.fuse3(down_inp=scale3_3, up_inp=up3,size=[inputs.shape[2],inputs.shape[3]], attention=attention3)
        fuse2 = self.fuse2(down_inp=scale2_2, up_inp=up2,size=[inputs.shape[2],inputs.shape[3]], attention=attention2)
        fuse1 = self.fuse1(down_inp=scale1_2, up_inp=up1,size=[inputs.shape[2],inputs.shape[3]], attention=attention1)



        # print(fuse5.shape)
        # print(fuse4.shape)
        # print(fuse3.shape)
        # print(fuse2.shape)
        # print(fuse1.shape)

        output = self.final(torch.cat([fuse5,fuse4,fuse3,fuse2,fuse1],1))

        # print(output.shape)
        return output, fuse5, fuse4, fuse3, fuse2, fuse1

if __name__ == '__main__':
    inp = torch.randn((1,3,512,512)).cuda()

    model = DeepCrack().cuda()

    out = model(inp)

    print("the end")
