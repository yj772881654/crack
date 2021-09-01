import Augmentor
import glob
import os
import random
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

train_path = '../datasets/train'
groud_truth_path = '../datasets/mask'
img_type = 'jpg'
train_tmp_path = '../datasets/tmp/train'
mask_tmp_path = '../datasets/tmp/mask'


def start(train_path, groud_truth_path):
    train_img = glob.glob(train_path + '/*.' + img_type)
    masks = glob.glob(groud_truth_path + '/*.' + "png")

    if len(train_img) != len(masks):
        print("trains can't match masks")
        return 0
    for i in range(len(train_img)):
        i = i + 1
        train_img_tmp_path = train_tmp_path + '/' + str(i)
        if not os.path.lexists(train_img_tmp_path):
            os.mkdir(train_img_tmp_path)
        img = load_img(train_path + '/' + str(i) + '.' + img_type)
        x_t = img_to_array(img)
        img_tmp = array_to_img(x_t)
        img_tmp.save(train_img_tmp_path + '/' + str(i) + '.' + img_type)

        mask_img_tmp_path = mask_tmp_path + '/' + str(i)
        if not os.path.lexists(mask_img_tmp_path):
            os.mkdir(mask_img_tmp_path)
        mask = load_img(groud_truth_path + '/' + str(i) + '.' + "png")
        x_l = img_to_array(mask)
        mask_tmp = array_to_img(x_l)
        mask_tmp.save(mask_img_tmp_path + '/' + str(i) + '.' + "png")
        print("%s folder has been created!" % str(i))
    return i + 1


def doAugment(num):
    sum = 0
    for i in range(num):
        p = Augmentor.Pipeline(train_tmp_path + '/' + str(i))
        p.ground_truth(mask_tmp_path + '/' + str(i))
        p.rotate(probability=0.5, max_left_rotation=5, max_right_rotation=5)  # 旋转
        p.flip_left_right(probability=0.5)  # 按概率左右翻转
        p.zoom_random(probability=0.6, percentage_area=0.99)  # 随即将一定比例面积的图形放大至全图
        p.flip_top_bottom(probability=0.6)  # 按概率随即上下翻转
        p.random_distortion(probability=0.8, grid_width=10, grid_height=10, magnitude=20)  # 小块变形
        count = random.randint(40, 60)
        print("\nNo.%s data is being augmented and %s data will be created" % (i, count))
        sum = sum + count
        p.sample(count)
        print("Done")
    print("%s pairs of data has been created totally" % sum)


a = start(train_path, groud_truth_path)
doAugment(a)