from tools.Crackloader import *

from tools.Make_testdata import *
from prf_metrics import cal_prf_metrics
import os
import glob
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import argparse

def make_dataset(root, crackNetName):
    # root = "./datasets/DeepCrack-DS/test/ "

    image_format = "*.jpg"
    lable_format = ".png"

    pred_imgs, gt_imgs = [], []


    images_base = os.path.join(root, crackNetName + "/test/")
    Lable_base = os.path.join(root, crackNetName + "/lable/")

    for pred_path in glob.glob(os.path.join(images_base,image_format)):


        gt_path = os.path.join(Lable_base, os.path.basename(pred_path)[:-4] + lable_format)

        # print(pred_path)
        # print(gt_path)

        gt_img = imread(gt_path, thresh=127)
        pred_img = imread(pred_path, gt_img)

        # print(gt_img.shape)
        gt_imgs.append(gt_img)
        pred_imgs.append(pred_img)

    return pred_imgs, gt_imgs

def imread(path, rgb2gray=None, load_size=0, load_mode=cv2.IMREAD_GRAYSCALE, convert_rgb=False, thresh=-1):
    im = cv2.imread(path, load_mode)
    # print("666666666666")
    # print(im.shape)
    if convert_rgb:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    if load_size > 0:
        im = cv2.resize(im, (load_size, load_size), interpolation=cv2.INTER_CUBIC)
    if thresh > 0:
        _, im = cv2.threshold(im, thresh, 255, cv2.THRESH_BINARY)
    else:
        # im = ((rgb2gray == 0) * 0.996+ (rgb2gray == 255) * 0.997) * im
        im = ((rgb2gray == 0) + (rgb2gray == 255) ) * im
    return im


def save_results(input_list, output_path, type):
    with codecs.open(output_path, 'w', encoding='utf-8') as fout:
        for ll in input_list:
            line = '\t'.join(['%.4f'%v for v in ll])+'\n'
            fout.write(line)

Test_path = "./draw_datasets/" # 将用来test的图片放到这个路径下

nets_list = os.listdir(Test_path)

# for CrackNetName in nets_list:
#
#     img_list, gt_list = make_dataset(Test_path, CrackNetName)
#     final_results_prf = cal_prf_metrics(img_list, gt_list, 0.01)
#     save_results(final_results_prf, './prfs/' + CrackNetName + '.prf' , "prf")


files = glob.glob(os.path.join("./prfs", "*.prf"))
_, axs = plt.subplots(nrows=1, ncols=1, figsize=(5,5))

files2 = ["/home/nlg/CrackNet/evalauation/prfs/Side1.prf",
          "/home/nlg/CrackNet/evalauation/prfs/Side2.prf",
          "/home/nlg/CrackNet/evalauation/prfs/Side3.prf",
          "/home/nlg/CrackNet/evalauation/prfs/Side4.prf",
          "/home/nlg/CrackNet/evalauation/prfs/Fused.prf"]
files3 = ["/home/nlg/CrackNet/evalauation/prfs/HED.prf",
          "/home/nlg/CrackNet/evalauation/prfs/OurProposed.prf",
          "/home/nlg/CrackNet/evalauation/prfs/DeepCrack.prf",
          "/home/nlg/CrackNet/evalauation/prfs/SegNet.prf",
          "/home/nlg/CrackNet/evalauation/prfs/U-Net.prf",
          "/home/nlg/CrackNet/evalauation/prfs/SRN.prf",
          "/home/nlg/CrackNet/evalauation/prfs/SE.prf","/home/nlg/CrackNet/evalauation/prfs/RCF.prf"]

for ff in files3:
    fname = ff.split('/')[-1].split('.')[0] # 每一个网络的 prf / aoc 文件
    p_acc, r_acc, f_acc = [], [], []
    with open(ff, 'r') as fin:
       for ll in fin:
           bt, p, r, f = ll.strip().split('\t')
           p_acc.append(float(p))
           r_acc.append(float(r))
           f_acc.append(float(f))
    # max_index = np.argmax(np.array(f_acc))
    # if fname=="HED":
    #     axs.plot(np.array(r_acc), np.array(p_acc),
    #              label='[F={:.03f}]{}'.format(0.719, fname).replace('=0.', '=.'), lw=1.25)
    # elif fname=="RCF":
    #     axs.plot(np.array(r_acc), np.array(p_acc), label='[F={:.03f}]{}'.format(0.789, fname).replace('=0.', '=.'), lw=1.25)
    # # axs.scatter(p_acc[max_index],r_acc[max_index])
    # elif fname == "SegNet":
    #     axs.plot(np.array(r_acc), np.array(p_acc),
    #              label='[F={:.03f}]{}'.format(0.794, fname).replace('=0.', '=.'), lw=1.25)
    # elif fname == "SRN":
    #     axs.plot(np.array(r_acc), np.array(p_acc),
    #              label='[F={:.03f}]{}'.format(0.735, fname).replace('=0.', '=.'), lw=1.25)
    # elif fname == "U-Net":
    #     axs.plot(np.array(r_acc), np.array(p_acc),
    #              label='[F={:.03f}]{}'.format(0.757, fname).replace('=0.', '=.'), lw=1.25)
    # elif fname == "OurProposed":
    #     axs.plot(np.array(r_acc), np.array(p_acc),
    #              label='[F={:.03f}]{}'.format(0.877, fname).replace('=0.', '=.'), lw=1.25)
    # elif fname == "SE":
    #     axs.plot(np.array(r_acc), np.array(p_acc),
    #              label='[F={:.03f}]{}'.format(0.557, fname).replace('=0.', '=.'), lw=1.25)
    # elif fname == "DeepCrack":
    #     axs.plot(np.array(r_acc), np.array(p_acc),
    #              label='[F={:.03f}]{}'.format(0.856, fname).replace('=0.', '=.'), lw=1.25)
    # if fname == "Method-two":
    #     axs.plot(np.array(r_acc), np.array(p_acc),
    #              label='[F={:.03f}]{}'.format(0.877, "Method-two").replace('=0.', '=.'), lw=1.25)
    # #
    # else:
    # axs.plot(np.array(r_acc), np.array(p_acc), label='[F={:.03f}]{}'.format(f_acc[max_index], fname).replace('=0.', '=.'), lw=1.25)

    axs.plot(r_acc[max_index],p_acc[max_index],"kx")

    axs.set_ylabel('precision')
    axs.set_xlabel('recall')

axs.grid(True, linestyle='-.')
axs.set_xlim([0., 1.])
axs.set_ylim([0., 1.])
axs.legend(loc='lower left')

pdf = PdfPages('prf.pdf')
plt.savefig(pdf, format='pdf', bbox_inches='tight')
pdf.close()
pdf=None





