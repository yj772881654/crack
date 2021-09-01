import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionBlock(nn.Module):
    def __init__(self, input_channels, output_channels, mode="sum", out="y", group=4):
        super(AttentionBlock, self).__init__()

        self.inter_channels = output_channels * 2
        self.mode = mode
        self.out = out
        self.group = group

        self.W_1 = nn.Sequential(
            nn.Conv2d(input_channels, self.inter_channels, kernel_size=3, stride=1, padding=1, bias=True)
            # nn.BatchNorm2d(inter_channels)
        )

        self.W_2 = nn.Sequential(
            nn.Conv2d(self.inter_channels, self.inter_channels, kernel_size=3, stride=1, padding=1, bias=True)
            # nn.BatchNorm2d(inter_channels)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(self.inter_channels, output_channels, kernel_size=3, stride=1, padding=1, bias=True),
            # nn.BatchNorm2d(output_channels),
        )

        self.norm_layer1 = nn.GroupNorm(self.group, self.inter_channels)
        self.norm_layer2 = nn.GroupNorm(self.group, self.inter_channels)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs, x):
        if self.mode == "sum":
            sum = 0
            for input in inputs:
               sum += input
        elif self.mode == "cat":
            sum = torch.cat(inputs, dim=1)

        out = self.relu(self.norm_layer1(self.W_1(sum)))
        out = self.relu(self.norm_layer2(self.W_2(out)))

        # psi = F.softmax(self.psi(out), dim=1) # Mask
        psi = F.sigmoid(self.psi(out))  # Mask

        if self.out == "n":
            out = x * psi
        elif self.out == "y":
            out = ((sum * psi).sum(1)).unsqueeze(1)
        return out


class SABlock(nn.Module):
    def __init__(self, input_channels, output_channels=1):
        super(SABlock, self).__init__()

        self.W_1 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=1, padding=1, bias=True),
            # nn.BatchNorm2d(output_channels)
        )

        # self.norm_layer = nn.GroupNorm(3, output_channels)
        self.sigmoid = nn.Sigmoid()
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, inputs, x):
        inputscat = torch.cat(inputs, dim=1)

        cat = self.W_1(inputscat)
        # cat = self.norm_layer(self.W_1(inputscat))
        psi = self.sigmoid(cat)
        # psi = self.softmax(cat)

        return x * psi

class FABlock(nn.Module):

    def __init__(self, input_channels, output_channels):
        super(FABlock, self).__init__()

        inter_channels = 32

        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, inter_channels, kernel_size=1, stride=1, padding=0, bias=True)
            # nn.BatchNorm2d(inter_channels)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(inter_channels, inter_channels, kernel_size=3, stride=1, padding=1, bias=True)
            # nn.BatchNorm2d(inter_channels)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(inter_channels, output_channels, kernel_size=1, stride=1, padding=0, bias=True)
            # nn.BatchNorm2d(output_channels),
            # nn.Softmax(dim=1)
        )

        self.norm_layer1 = nn.GroupNorm(4, inter_channels)
        self.norm_layer2 = nn.GroupNorm(4, inter_channels)

        # self.fuse = nn.Conv2d(output_channels, 1, kernel_size=1, stride=1, padding=0, bias=True)

        self.relu = nn.ReLU()

    def forward(self, fusecat):
        fusecat = torch.cat(fusecat, dim=1)
        attn = self.relu(self.norm_layer1(self.conv1(fusecat)))
        attn = self.relu(self.norm_layer2(self.conv2(attn)))
        attn = F.softmax(self.conv3(attn), dim=1)

        return ((fusecat * attn).sum(1)).unsqueeze(1) # self.fuse(fusecat * attn) #