import os


datasetName = "DeepCrack-DS"

img_aug_dir = "../datasets/" + datasetName + "/train/Train_image_aug/"
img_aug_dir_temp = "../datasets/" + datasetName + "/train/Train_image_aug/p1/" #因为每个文件夹下的图像名字都一样，所以我们只用获取其中一个就可以了
img_list = os.listdir(img_aug_dir_temp)
file_list = os.listdir(img_aug_dir)
listTXT = "../datasets/" + datasetName + "/train/DeepCrack-DS_aug.txt"# + datasetName + ".txt"

try:
    if os.path.exists(listTXT) == False:
        os.makedirs(listTXT)
except:
    print("创建文件失败")

for file_name in file_list:
        for imageName in img_list:
            first = '/home/nlg/CrackNet/datasets/DeepCrack-DS/train/Train_image_aug/' + file_name + '/' + imageName
            second = '/home/nlg/CrackNet/datasets/DeepCrack-DS/train/Lable_image_aug/' + file_name + '/' + imageName[:-4] + ".png"
            with open(listTXT, 'a', encoding='utf-8') as fout:
                line =  first + ' ' + second + '\n'
                fout.write(line)

# for imageName in img_list:
#     first = '/home/nlg/CrackNet/datasets/DeepCrack-DS/train/Train_image_aug/' + imageName
#     second = '/home/nlg/CrackNet/datasets/DeepCrack-DS/train/Lable_image_aug/' + imageName[:-4] + ".png"
#     with open(listTXT, 'a', encoding='utf-8') as fout:
#         line =  first + ' ' + second + '\n'
#         fout.write(line)