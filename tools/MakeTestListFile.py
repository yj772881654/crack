# img_aug_dir = "../datasets/CFD/train/Train_image_aug/"
import os

# img_aug_dir = "/home/nlg/CrackNet/datasets/DeepCrack-DS/test/Test_image/"
import cv2

img_aug_dir = "/home/nlg/DeepCrack-master/codes/data/Stone/test/"
# lab_aug_dir = "./datasets/CFD/train/Lable_image_aug/"

# img_aug_dir_temp = "../datasets/CFD/train/Train_image_aug/p1/" #因为每个文件夹下的图像名字都一样，所以我们只用获取其中一个就可以了
# img_aug_dir_temp = "/home/nlg/CrackNet/datasets/DeepCrack-DS/test/Test_image/" #因为每个文件夹下的图像名字都一样，所以我们只用获取其中一个就可以了
img_aug_dir_temp = "/home/nlg/DeepCrack-master/codes/data/Stone/test-gt/"

img_list = os.listdir(img_aug_dir_temp)
file_list = os.listdir(img_aug_dir)


for file_name in file_list:
        # for imageName in img_list:
            # first = 'Train_image_aug/'+ file_name +'/' + imageName
            # second = 'Lable_image_aug/' + file_name +'/' + imageName[:-4] + ".png"
            # first = '/home/nlg/CrackNet/datasets/DeepCrack-DS/test/Test_image/' + file_name
            # second = '/home/nlg/CrackNet/datasets/DeepCrack-DS/test/Lable_image/'+ file_name[:-4] + ".png"
            first = '/home/nlg/DeepCrack-master/codes/data/Stone/test/' + file_name
            second = '/home/nlg/DeepCrack-master/codes/data/Stone/test-gt/' + file_name[:-4] + ".bmp"

            im = cv2.imread(first)
            gt = cv2.imread(second)
            im_cropped = im[256:768,256:768]
            gt = cv2.resize(gt, (1024, 1024))
            gt_cropped = gt[256:768,256:768]

            cv2.imwrite(first,im_cropped)
            cv2.imwrite(second, gt_cropped)
            # with open("../datasets/CFD/train/CFD.txt", 'a', encoding='utf-8') as fout:
            # with open("../datasets/Stone.txt", 'a', encoding='utf-8') as fout:
            #     line =  first + ' ' + second + '\n'
            #     fout.write(line)
