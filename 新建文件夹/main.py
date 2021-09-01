import os
import cv2
import numpy as np
Cal_Ods=False
def get_statistics( pred, gt):
    """
    return tp, fp, fn
    """
    tp = np.sum((pred == 1) & (gt == 1))
    fp = np.sum((pred == 1) & (gt == 0))
    fn = np.sum((pred == 0) & (gt == 1))
    return [tp, fp, fn]
def cal_ois_metrics(pred, gt, thresh_step=0.01):
    statistics = []
    for thresh in np.arange(0.0, 1.0, thresh_step):
        gt_img = (gt / 255).astype('uint8')
        pred_img = (pred / 255 > thresh).astype('uint8')
        tp, fp, fn = get_statistics(pred_img, gt_img)
        p_acc = 1.0 if tp == 0 and fp == 0 else tp / (tp + fp)
        r_acc = tp / (tp + fn)

        if p_acc + r_acc == 0:
            f1 = 0
        else:
            f1 = 2 * p_acc * r_acc / (p_acc + r_acc)
        statistics.append([thresh, f1])
    max_f = np.amax(statistics, axis=0)
    return max_f
def cal_dir_ois_metrics(pred_list, gt_list, thresh_step=0.01):
    final_acc_all = []
    for pred, gt in zip(pred_list, gt_list):
        statistics = []
        for thresh in np.arange(0.0, 1.0, thresh_step):
            gt_img = (gt / 255).astype('uint8')
            pred_img = (pred / 255 > thresh).astype('uint8')
            tp, fp, fn = get_statistics(pred_img, gt_img)
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

ls=os.listdir()
lbl=[]
img=[]
cv2_img=[]
cv2_lbl=[]
for each in ls:
    if each.endswith(".bmp"):
        lbl.append(each)
        cv2_lbl.append(cv2.imread(each))
    if each.endswith(".jpg"):
        img.append(each)
        cv2_img.append(cv2.imread(each))
f=open("file.txt",'w')
for i in range(len(img)):
    Img=cv2.imread(img[i])
    Lbl=cv2.imread(lbl[i])
    ans=cal_ois_metrics(Img,Lbl)
    print(lbl[i],ans[1])
    f.writelines(str(img[i])+" "+str(ans[1])+'\n')

if Cal_Ods:
    ans=cal_dir_ois_metrics(cv2_img,cv2_lbl)
    print(ans)
    f.writelines(str(ans))

f.close()

# import os
# import shutil
# ls=[]
# with open('file.txt') as f:
#     str=f.readlines()
#     f.close()
# for each in str:
#     name=each.strip().split(' ')[0]
#     num=each.strip().split(' ')[1]
#     # print(name,num)
#     ls.append((name,float(num)))
# ls=sorted(ls,key=lambda ois:ois[1],reverse=True)
# for each in ls:
#     print(each)
# for each in range(20):
#     name=ls[each][0]
#     shutil.move(name,'img/'+name)
#     shutil.move(name[:-4]+'.bmp','gt/'+name)