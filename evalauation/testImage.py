import time

from nets.Multi_Loss_FusionNet import *
from tools.Crackloader import *
from torch.autograd import Variable
import torch
from PIL import Image
import os
import torch.nn.functional as F
# import cv2
from tools.Make_testdata import *
from prf_metrics import cal_prf_metrics
from segment_metrics import cal_semantic_metrics
from roc_metrics import  cal_roc_metrics
from calculate_OIS import cal_ois_metrics
from tqdm import tqdm

model_root = '../model/epoch1.pkl'  # 模型
# model_root = '../model/ResFusion_all/best_ResFusion_all8515.pkl'  # 模型

# datasetName = "CFD"
# datasetName = "CrackTree"
# datasetName = "CRKWH100"
# datasetName = "CrackLS315"
datasetName = "CrackLS315"
# datasetName = "Crack500"

Test_path = "../datasets/" + datasetName + "/test/" # 将用来test的图片放到这个路径下
CrackNetName = "CANet"
createPred = True
calculate = True
# img_transforms = transforms.Compose(
#             [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
img_transforms = transforms.Compose(
            [transforms.ToTensor()])


crack = torch.load(model_root )

Image_path = Test_path + "Test_image/"  # 遍历该文件下的每张图片
image_list = os.listdir(Image_path)
i = 0
if createPred:
    print("生成预测结果图。。。。。。。。。。")
    # for image_name in tqdm(image_list):
    for image_name in image_list:
        # print(image_name)

        image = os.path.join(Image_path, image_name)

        image = Image.open(image)
        # print(type(image))
        # print(image_name)
        x = Variable(img_transforms(image)).cuda()
        x = x.unsqueeze(0)
        # crack.eval()            # 取消掉dropout
        y = crack.forward(x)    # 前向传播，得到处理后的图像y（tensor形式）
        output = torch.sigmoid(y[-1])
        i += 1
        print("生成第{}张图片中".format(i))

        # print(np.max(output.clone().cpu().detach().numpy()))
        # print(np.min(output.clone().cpu().detach().numpy()))
        out_clone = output.clone().cpu().data.numpy()
        out_clone = np.squeeze(out_clone, axis=0)
        out_clone = np.transpose(out_clone, (1, 2, 0))

        test_res = "{}{}_result/".format(Test_path, CrackNetName)
        try:
            if os.path.exists(test_res) == False:
                os.makedirs(test_res)
        except:
            print("创建valid_res文件失败")

        gt_path = os.path.join(Test_path + "Lable_image/", os.path.basename(image_name)[:-4] + ".bmp")
        gt_img = cv2.imread(gt_path)
        out_clone = ((gt_img == 0)  + (gt_img == 255) ) * out_clone
        cv2.imwrite(test_res + image_name, out_clone * 255.0)

#################################################################################
if calculate:

    img_list, gt_list = make_test_dataset(Test_path, CrackNetName, datasetName)


    # print("测算 pre recall f 中。。。。")
    final_results_prf = cal_prf_metrics(img_list, gt_list, 0.01)
    max_f = np.amax(final_results_prf, axis=0)

    # print("测算 roc 中。。。。")
    final_results_roc = cal_roc_metrics(img_list, gt_list, 0.01)
    max_roc = np.amax(final_results_roc, axis=0)

    print("测算 mIOU 中。。。。")
    final_results_metrics = cal_semantic_metrics(img_list, gt_list, 0.01)
    max_metrics = np.amax(final_results_metrics, axis=0)

    print("测算 OIS/ODS 中。。。。")
    final_results_OIS= cal_ois_metrics(img_list, gt_list, 0.01)

    with open("./TestReport.txt", 'a', encoding='utf-8') as fout:

        line1 = "===================测试结果如下=========================\n"
        line2 =  "测试网络名称：" + CrackNetName
        line3 = "测试时间：" + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
        line4 = "测试的数据集：{}".format(Test_path.split('/')[2])
        line5 = "测试数据集的大小：{}张图片".format(len(img_list))
        line6 = "测试数据集的ODS : {:.4f} / OIS : {:.4f} / mIOU : {:.4f}".format(max_f[3],final_results_OIS,max_metrics[3])
        # line7 = "测试数据集的gACC:{:.6f}/mACC:{:.6f}/mIOU:{:.6f}".format(max_metrics[1],max_metrics[2],max_metrics[3])

        fout.write(line1 + "\n")
        fout.write(line2 + "\n")
        fout.write(line3 + "\n")
        fout.write(line4 + "\n")
        fout.write(line5 + "\n")
        fout.write(line6 + "\n")
        # fout.write(line7 + "\n")

        print(line1)
        print(line2)
        print(line3)
        print(line4)
        print(line5)
        print(line6)
        # print(line7)


        # print("测试数据集的pre:{:.6f}/recall:{:.6f}".format(max_f[1],max_f[2]))
        # print("测试数据集的Sens.:{:.6f}/Spec:{:.6f}".format(max_roc[1],max_roc[3]))

    save_results(final_results_prf, './evalData/' + CrackNetName + '.prf' , "prf")
    save_results(final_results_roc, './evalData/' + CrackNetName + '.roc', "roc")
    save_results(final_results_metrics, './evalData/' + CrackNetName + '.sem', "sem")







