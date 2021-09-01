import codecs
import glob
import os
import numpy as np
import cv2


def make_test_dataset(root, crackNetName, datasetName):
    # root = "./datasets/DeepCrack-DS/test/ "

    image_format = "*.jpg"
    lable_format = ".png"
    if datasetName == "CrackLS315":
        image_format = "*.jpg"
        lable_format = ".bmp"
    elif datasetName == "CRKWH100":
        image_format = "*.bmp"
        lable_format = ".bmp"
    elif datasetName == "CrackTree":
        image_format = "*.jpg"
        lable_format = ".bmp"
    elif datasetName == "Crack500":
        image_format = "*.jpg"
        lable_format = ".png"


    pred_imgs, gt_imgs = [], []


    images_base = os.path.join(root, crackNetName+'_result/')
    Lable_base = os.path.join(root, "Lable_image")

    for pred_path in glob.glob(os.path.join(images_base,image_format)):

        gt_path = os.path.join(Lable_base, os.path.basename(pred_path)[:-4] + lable_format)

        gt_img = imread(gt_path, thresh=127)
        pred_img = imread(pred_path, gt_img)

        # print(gt_img.shape)
        gt_imgs.append(gt_img)
        pred_imgs.append(pred_img)

    return pred_imgs, gt_imgs



def imread(path, rgb2gray=None, load_size=0, load_mode=cv2.IMREAD_GRAYSCALE, convert_rgb=False, thresh=-1):
    im = cv2.imread(path, load_mode)
    if convert_rgb:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    if load_size > 0:
        im = cv2.resize(im, (load_size, load_size), interpolation=cv2.INTER_CUBIC)
    if thresh > 0:
        _, im = cv2.threshold(im, thresh, 255, cv2.THRESH_BINARY)
    else:
        im = ((rgb2gray == 0) * 0.5 + (rgb2gray == 255) * 1) * im
    return im

def save_results(input_list, output_path, type):
    with codecs.open(output_path, 'w', encoding='utf-8') as fout:
        for ll in input_list:
            line = '\t'.join(['%.4f'%v for v in ll])+'\n'
            fout.write(line)
        idx = np.argmax(input_list, axis=0) #每一列中最大的那个值的坐标
        eval_data = { 'roc': ['','Sensitivity','fpr','Specificity'],
                      'prf':['','precision','recall','f-measure'],
                      'sem':['','global_acc', 'mean_acc', 'mean_iou_acc']
                    }

        for i in range(1,4):
            line1 = 'max_' + eval_data[type][i] +'\n'
            fout.write(line1)
            line2 = '\t'.join(['%.4f' % v for v in input_list[idx[i]]])+'\n'
            fout.write(line2)