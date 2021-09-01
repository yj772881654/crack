import os
import os.path
import torch
from torchvision import transforms
import numpy as np
import scipy.misc as m
import glob
import torch.utils.data as data

from torch.utils import data

class Crackloader(data.Dataset):
 

    def __init__(self,root,transform=None):
        
        self.root = root
        # self.transform = transform
        self.train_set_path = self.make_dataset(root)
        self.img_transforms = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    def __len__(self):
        
        return len(self.train_set_path)

    def __getitem__(self, index):
        
        img_path,lbl_path = self.train_set_path[index]

        img = m.imread(img_path, mode='RGB')
        img = np.array(img, dtype=np.uint8)
        img = transforms.ToTensor()(img)
        # img = self.img_transforms(img)
        # print("img:{}".format(img.dtype))
        lbl = m.imread(lbl_path)

        lbl = np.array(lbl, dtype=np.uint8)

        lbl = np.expand_dims(lbl, axis= 0)
        # lbl = transforms.ToTensor()(lbl)
       
        lbl = torch.FloatTensor(lbl/255)
        # print("lable:{}".format(lbl.dtype))
        # if not self.transform:
        #     img, lbl = self.transform(img, lbl)

        return img, lbl

    

    def make_dataset(self,root):
        dataset = []
        images_base = os.path.join(root, "Train_image")
        Lable_base = os.path.join(root, "Lable_image")

        for img_path in glob.glob(os.path.join(images_base,"*.jpg")):
            lb_path = os.path.join(Lable_base, os.path.basename(img_path)[:-4] + ".png")
            dataset.append([img_path,lb_path])
        return dataset
		



