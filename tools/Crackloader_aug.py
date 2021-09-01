import os
import os.path
import torch
from torchvision import transforms
import numpy as np
import scipy.misc as m
import glob
import torch.utils.data as data
import cv2
from torch.utils import data


class Crackloader_aug(data.Dataset):

    def __init__(self, root, transform=None):
        self.root = root
        self.img_transforms = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.train_set_path = self.make_dataset(root)

    def __len__(self):
        return len(self.train_set_path)

    def __getitem__(self, index):
        img_path, lbl_path = self.train_set_path[index]

        # img = m.imread(self.root + img_path, mode='RGB')
        img = cv2.imread(self.root + img_path, cv2.IMREAD_UNCHANGED)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = np.array(img, dtype=np.uint8)
        img = self.img_transforms(img)

        # img = transforms.ToTensor()(img)
        # print("img:{}".format(img.dtype))

        lbl = cv2.imread(self.root + lbl_path)

        lbl = np.array(lbl, dtype=np.uint8)
        lbl = lbl[:,:,1]
        lbl = np.expand_dims(lbl, axis=0)
        # lbl = transforms.ToTensor()(lbl)


        _, lbl = cv2.threshold(lbl, 127, 1, cv2.THRESH_BINARY)



        return img, lbl

    def make_dataset(self, root):
        # root = "../datasets/DeepCrack-DS/train/"
        datasetName = root.split("/")[2]
        dataset = []
        datasetTxt = os.path.join(root, datasetName+'.txt')
        
        with open(datasetTxt, 'r') as f:
            for line in f.readlines():
               line = ''.join(line).strip('\n')
               line_list =  line.split(' ')
               dataset.append([line_list[0], line_list[1]])
        return dataset




