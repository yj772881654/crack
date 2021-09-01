import shutil
from PIL import Image
from nets.Multi_Loss_FusionNet import *
import torch.nn as nn
from torch.autograd import Variable

from new_net.LABlock import DeepCrack
# from nets.deepcrack3_lam2 import DeepCrack
from new_net.deepcrack_pvt import DeepCrack
from new_net.LABlock import DeepCrack
from new_net.LABlock_replace import DeepCrack
# from nets.deepcrack3_lablock import DeepCrack
from utils.utils import *
from utils.Validator import *
from utils.Crackloader import *
import cv2

def trainer(net, total_epoch, lr_init, batch_size,train_img_dir, valid_img_dir, valid_lab_dir,
            valid_result_dir, valid_log_dir, best_model_dir, image_format, lable_format, pretrain_dir=None):

    # 输入训练图片,制作成dataset,dataset其实就是一个List 结构如同：[[img1,gt1],[img2,gt2],[img3,gt3]]
    # img_data = Crackloader(root=img_dir)

    img_data = Crackloader(txt_path=train_img_dir, normalize=False) #如果做了数据增强，就用这个加载器

    #  加载数据 用pytorch给我们设计的加载器，可以自动打乱
    img_batch = data.DataLoader(img_data, batch_size=batch_size, shuffle=True, num_workers=2)

    print(img_batch)
    # 这是根据机器有几块GPU进行选择
    if torch.cuda.device_count() > 1:
        crack = nn.DataParallel(net).cuda()
    else:
        crack = net.cuda()  # 如果没有多块GPU需要用这种方法

    # 可以加载训练好的模型
    if pretrain_dir is not None:
        crack = torch.load(pretrain_dir).cuda()

    # 生成验证器
    validator = Validator(valid_img_dir, valid_lab_dir,
                          valid_result_dir, valid_log_dir, best_model_dir, crack,  image_format, lable_format)
    # 训练的核心部分

    for epoch in range(1, total_epoch):

        losses = Averagvalue()

        crack.train()  # 选择训练状态

        count = 0 # 记录在当前epoch下第几个item，即当前epoch下第几次更新参数

        # 关于学习率更新的办法
        new_lr = updateLR(lr_init, epoch, total_epoch)

        print('Learning Rate: {:.9f}'.format(new_lr))

        # optimizer
        lr = new_lr
        optimizer = torch.optim.Adam(crack.parameters(), lr=lr)

        for (images, labels) in img_batch:
            count += 1
            loss = 0

            # 前向传播部分
            images = Variable(images).cuda()
            labels = Variable (labels.float()).cuda()

            output = crack.forward(images)
            for out in output:
                loss += 0.5 * crack.calculate_loss(out, labels)
            loss += 1.1 * crack.calculate_loss(output[-1], labels)
            #计算损失部分
            optimizer.zero_grad()

            loss.backward()
            optimizer.step()

            #打印输出损失，lr等
            losses.update(loss.item(), images.size(0))
            lr = optimizer.param_groups[0]['lr']
            if count%10==0:
                info = 'Epoch: [{0}/{1}][{2}/{3}] '.format(epoch, total_epoch, count, len(img_batch)) + \
                       'Loss {loss.val:f} (avg:{loss.avg:f} lr {lr:.10f}) '.format(
                           loss=losses, lr=lr)

                print(info)
            # validator.validate(i)
            if count%600==0:
                print("test valid")
                validator.validate(epoch)
                crack.train()
            # torch.save(crack, "./model/epoch{}.pkl".format(str(i)))  # 保存网络
        # if (i+1) % 2 == 0:
        if epoch % 2 == 0:
            print("test valid")
            validator.validate(epoch)
            # torch.save(crack, "./model/epoch{}.pkl".format(str(epoch)))

        # optimizer = torch.optim.Adam(crack.parameters(), lr=lr)
        # print("学习率更新为：{:.6f}".format(lr))


if __name__ == '__main__':

    # datasetName = "CrackTree"
    datasetName="CrackLS315"
    # datasetName="100"
    # dataName = "train_aug_ads"
    dataName= "train_ads"
    netName = "CANet_New"
    # 
    image_format = "jpg"
    lable_format = "bmp"

    total_epoch = 300# 训练步数
    lr_init = 0.001    # 定义学习率
    batch_size = 1      # mini batch 大小

    # net = CANetGenerator(backbone='resnext50', rate=4)
    net = DeepCrack()

    train_img_dir = "./datasets/"+ datasetName +"/train/" + dataName +".txt"
    valid_img_dir = "./datasets/"+datasetName+"/valid/Valid_image/"
    valid_lab_dir = "./datasets/"+datasetName+"/valid/Lable_image/"
    valid_result_dir = "./datasets/"+datasetName+"/valid/Valid_result/"

    valid_log_dir = "./log/" + netName 
    best_model_dir = "./model/" + datasetName +"/"
    # pretrain_dir="/home/nlg/yj/mxy/code/crack/model/0.703.pkl"
    trainer(net, total_epoch, lr_init, batch_size, train_img_dir, valid_img_dir, valid_lab_dir,
            valid_result_dir, valid_log_dir, best_model_dir,  image_format, lable_format) #, pretrain_dir=pretrain_dir

    print("训练结束")

