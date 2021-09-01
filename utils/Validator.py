import cv2
import os.path
import torch
from torchvision import transforms
import numpy as np
import scipy.misc as m
import glob
import torch.nn as nn
from torch.autograd import Variable

class Validator(object):

    def __init__(self, valid_img_dir, valid_lab_dir, valid_result_dir, valid_log_dir, best_model_dir,
                 net,image_format = "jpg",lable_format = "png", normalize = False):

        self.valid_img_dir = valid_img_dir  # 验证集的路径
        self.valid_lab_dir = valid_lab_dir  # 验证集GT的路径
        self.valid_res_dir = valid_result_dir # 验证集生成结果的路径
        self.best_model_dir = best_model_dir
        self.valid_log_dir = valid_log_dir + "/valid.txt" # 验证集测试指标的路径
        self.image_format = image_format
        self.lable_format = lable_format
        self.best_model_dir
        self.ods = 0

        self.net = net
        self.normalize = normalize
        # 数值归一化到[-1, 1]
        if self.normalize:
            self.transforms = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        else:
            self.transforms = transforms.ToTensor()

        try:
            if not os.path.exists(self.valid_log_dir):
                os.makedirs(self.valid_log_dir)
        except:
            print("创建valid_log文件失败")

    def make_dir(self):
        try:
            if not os.path.exists(self.valid_res_dir):
                os.makedirs(self.valid_res_dir)
        except:
            print("创建valid_res文件失败")

        try:
            if not os.path.exists(self.valid_log_dir):
                os.makedirs(self.valid_log_dir)
        except:
            print("创建valid_log文件失败")

    def make_dataset(self, epoch_num):
        pred_imgs, gt_imgs = [], []
        for pred_path in glob.glob(os.path.join(self.valid_res_dir + str(epoch_num) + "/", "*." + self.image_format)):

            gt_path = os.path.join(self.valid_lab_dir, os.path.basename(pred_path)[:-4] + "." + self.lable_format)

            gt_img = self.imread(gt_path, thresh=80)
            # gt_img=cv2.resize(gt_img,(512,512))
            pred_img = self.imread(pred_path, gt_img)
            # pred_img=cv2.resize(pred_img,(512,512))
            gt_imgs.append(gt_img)
            pred_imgs.append(pred_img)
        # print(len(gt_imgs))
        # print(len(pred_imgs))

        return pred_imgs, gt_imgs

    def imread(self, path, rgb2gray=None, load_size=0, load_mode=cv2.IMREAD_GRAYSCALE, convert_rgb=False, thresh=-1):
        im = cv2.imread(path, load_mode)
        # im=cv2.resize(im,(512,512))
        # im = cv2.resize(im, (400, 400))

        if convert_rgb:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        if load_size > 0:
            im = cv2.resize(im, (load_size, load_size), interpolation=cv2.INTER_CUBIC)
        if thresh > 0:
            _, im = cv2.threshold(im, thresh, 255, cv2.THRESH_BINARY)
        else:
            im = ((rgb2gray == 0) + (rgb2gray == 255)) * im
        return im

    def get_statistics(self, pred, gt):
        """
        return tp, fp, fn
        """
        tp = np.sum((pred == 1) & (gt == 1))
        fp = np.sum((pred == 1) & (gt == 0))
        fn = np.sum((pred == 0) & (gt == 1))
        return [tp, fp, fn]

    # 计算 ODS 方法
    def cal_prf_metrics(self, pred_list, gt_list, thresh_step=0.01):
        final_accuracy_all = []
        for thresh in np.arange(0.0, 1.0, thresh_step):
            # print(thresh,end="")
            statistics = []

            for pred, gt in zip(pred_list, gt_list):
                
                gt_img = (gt / 255).astype('uint8')
                pred_img = ((pred / 255) > thresh).astype('uint8')
                # calculate each image
                statistics.append(self.get_statistics(pred_img, gt_img))

            # get tp, fp, fn
            tp = np.sum([v[0] for v in statistics])
            fp = np.sum([v[1] for v in statistics])
            fn = np.sum([v[2] for v in statistics])

            # calculate precision
            p_acc = 1.0 if tp == 0 and fp == 0 else tp / (tp + fp)
            # calculate recall
            r_acc = tp / (tp + fn)
            # calculate f-score
            final_accuracy_all.append([thresh, p_acc, r_acc, 2 * p_acc * r_acc / (p_acc + r_acc)])

        return final_accuracy_all

    # 计算 OIS 方法
    def cal_ois_metrics(self,pred_list, gt_list, thresh_step=0.01):
        final_acc_all = []
        for pred, gt in zip(pred_list, gt_list):
            statistics = []
            for thresh in np.arange(0.0, 1.0, thresh_step):
                gt_img = (gt / 255).astype('uint8')
                pred_img = (pred / 255 > thresh).astype('uint8')
                tp, fp, fn = self.get_statistics(pred_img, gt_img)
                p_acc = 1.0 if tp == 0 and fp == 0 else tp / (tp + fp)
                r_acc = tp / (tp + fn)

                if p_acc + r_acc == 0:
                    f1 = 0
                else:
                    f1 = 2 * p_acc * r_acc / (p_acc + r_acc)
                statistics.append([thresh, f1])
            max_f = np.amax(statistics, axis=0)
            final_acc_all.append(max_f[1])
        return np.mean(final_acc_all)
        # final_accuracy_all = []
        # for pred, gt in zip(pred_list, gt_list):
        #     statistics = []
        #     for thresh in np.arange(0.0, 1.0, thresh_step):
        #         gt_img = (gt / 255).astype('uint8')
        #         pred_img = (pred / 255 > thresh).astype('uint8')
        #         tp,fp,fn = get_statistics(pred_img, gt_img)
        #         p_acc = 1.0 if tp == 0 and fp == 0 else tp / (tp + fp)
        #         r_acc = tp / (tp + fn)
        #         # print(r_acc+ p_acc)
        #         # print(2 * p_acc * r_acc / (p_acc + r_acc))
        #         if p_acc + r_acc == 0:
        #             f1 = 0
        #         else:
        #             f1 = 2 * p_acc * r_acc / (p_acc + r_acc)
        #         statistics.append([thresh,f1])
        #     max_f = np.amax(statistics, axis=0)
        #     final_accuracy_all.append(max_f[1])
        # return np.mean(final_accuracy_all)
    # 如果需要计算指标，在此处添加计算方法即可


    def validate(self, epoch_num):
        print('开始验证，请您稍等。。。。。。。。')
        image_list = os.listdir(self.valid_img_dir)

        try:
            if not os.path.exists(self.valid_res_dir + str(epoch_num) + '/'):
                os.makedirs(self.valid_res_dir + str(epoch_num) + '/')
        except:
            print("创建valid_res文件失败")
        # 遍历该文件下的每张图片

        self.net.eval()  # 取消掉dropout
        with torch.no_grad():
            for image_name in image_list:
                image = os.path.join(self.valid_img_dir, image_name)

                image = cv2.imread(image)
                x = Variable(self.transforms(image)).cuda()
                x = x.unsqueeze(0)

                outs = self.net.forward(x)  # 前向传播，得到处理后的图像y（tensor形式）
                # if self.normalize:
                #     # 放缩到 [-1,1]
                #     output = torch.sigmoid(y[-1]) - 0.5 / 0.5
                # else:
                # output = torch.sigmoid(y[-1])
                y = outs[-1]
                output = torch.sigmoid(y)
                # output = y[-1]
                out_clone = output.clone()
                img_fused = np.squeeze(out_clone.cpu().detach().numpy(), axis=0)

                img_fused = np.transpose(img_fused, (1, 2, 0))
                # print(self.valid_res_dir  + str(epoch_num) + '/' + image_name)
                cv2.imwrite(self.valid_res_dir  + str(epoch_num) + '/' + image_name, img_fused * 255.0)



