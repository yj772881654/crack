import cv2
import os.path
import torch
from torchvision import transforms
import numpy as np
import scipy.misc as m
import glob
import torch.nn as nn
from torch.autograd import Variable

print('开始验证，请您稍等。。。。。。。。')


class Tester(object):

    def __init__(self, test_img_dir, test_lab_dir, test_result_dir, test_log_dir,
                 net, normalize = False,image_format = "jpg",lable_format = "png"):


        self.valid_img_dir = test_img_dir  # 验证集的路径
        self.valid_lab_dir = test_lab_dir  # 验证集GT的路径
        self.valid_res_dir = test_result_dir # 验证集生成结果的路径
        self.best_model_dir = best_model_dir
        self.valid_log_dir = valid_log_dir + "/valid.txt" # 验证集测试指标的路径
        self.image_format = image_format
        self.lable_format = lable_format

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

            gt_img = self.imread(gt_path, thresh=127)
            pred_img = self.imread(pred_path, gt_img)

            gt_imgs.append(gt_img)
            pred_imgs.append(pred_img)
        # print(len(gt_imgs))
        # print(len(pred_imgs))

        return pred_imgs, gt_imgs

    def imread(self, path, rgb2gray=None, load_size=0, load_mode=cv2.IMREAD_GRAYSCALE, convert_rgb=False, thresh=-1):
        im = cv2.imread(path, load_mode)
        if convert_rgb:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        if load_size > 0:
            im = cv2.resize(im, (load_size, load_size), interpolation=cv2.INTER_CUBIC)
        if thresh > 0:
            _, im = cv2.threshold(im, thresh, 255, cv2.THRESH_BINARY)
        else:
            im = ((rgb2gray == 0) + (rgb2gray == 255)) * im
        return im

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

    def get_statistics(self, pred, gt):
        """
        return tp, fp, fn
        """
        tp = np.sum((pred == 1) & (gt == 1))
        fp = np.sum((pred == 1) & (gt == 0))
        fn = np.sum((pred == 0) & (gt == 1))
        return [tp, fp, fn]

    def test(self,epoch_num):

        # 对生成的预测结果进行指标测量
        img_list, gt_list = self.make_dataset(epoch_num)

        # 计算prf
        final_results = self.cal_prf_metrics(img_list, gt_list, 0.01)
        max_f = np.amax(final_results, axis=0)

        with open(self.test_log_dir, 'a', encoding='utf-8') as fout:
            line = "epoch:{} | ODS:{:.6f}".format(epoch_num, max_f[3]) + '\n'
            fout.write(line)
        print("epoch={} 此时最大F值为：{:.6f}".format(epoch_num, max_f[3]))

        if max_f[3] >= self.ods:
            self.ods = max_f[3]

            # ods_str = str(ods)[2:6]
            # torch.save(net, self.best_model_dir + ods_str + ".pkl")

        print('验证完成，继续进行训练^_^')

    def generate_images(self):
        print('开始验证，请您稍等。。。。。。。。')
        image_list = os.listdir(self.valid_img_dir)
        self.net.eval()  # 取消掉dropout
        with torch.no_grad():
            for image_name in image_list:
                image_dir = os.path.join("../datasets/DeepCrack-DS/test/Test_image/", image_name)

                image = cv2.imread(image_dir)
                x = Variable(transforms.ToTensor()(image)).cuda()
                x = x.unsqueeze(0)
                net = torch.load("../model/epoch900.pkl").cuda()
                net.eval()  # 取消掉dropout
                y = net.forward(x)  # 前向传播，得到处理后的图像y（tensor形式）
                # if self.normalize:
                #     # 放缩到 [-1,1]
                #     output = torch.sigmoid(y[-1]) - 0.5 / 0.5
                # else:
                output = torch.sigmoid(y[-1])
                # print(output)
                out_clone = output.clone()
                img_fused = np.squeeze(out_clone.cpu().detach().numpy(), axis=0)

                img_fused = np.transpose(img_fused, (1, 2, 0))
                # print(image_dir)
                cv2.imwrite("../datasets/DeepCrack-DS/test/Test_result/" + image_name, img_fused * 255)

