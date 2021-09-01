import cv2
import numpy as np
import matplotlib.pyplot as plt

#
imgName = "774"
im_lab = cv2.imread("/home/nlg/CrackNet/tools/copy/Predicted/"+imgName+".bmp")
im_m1 = cv2.imread("/home/nlg/CrackNet/tools/copy/Predicted/"+imgName+".bmp")
im_m2 = cv2.imread("/home/nlg/CrackNet/tools/copy/Original/"+imgName+".jpg")  #input
#
# imgName = "0063-1"
# im_lab = cv2.imread("/home/nlg/CrackNet/tools/result/1008.bmp")
# im_m1 = cv2.imread("/home/nlg/CrackNet/tools/result/1008.bmp")
# im_m2 = cv2.imread("/home/nlg/CrackNet/tools/result/1008.png")
#
# imgName = "1008"
# im_lab = cv2.imread("/home/nlg/results/"+imgName+".bmp")
# im_m1 = cv2.imread("/home/nlg/results/"+imgName+".bmp")
# im_m2 = cv2.imread("/home/nlg/results/"+imgName+".png")

t = 0
max = 0
for i in range(513,im_lab.shape[0]):
    sum = 0
    for j in range(0,im_lab.shape[1]):
        sum += im_lab[i][j][0]
    if sum > max:
        max = sum
        t = i
print(t)
x = range(im_lab.shape[1])

t = 512+250




la = im_lab[:,:,2]
lm1 = im_m1[:,:,2]
lm2 = im_m2[:,:,2]



A, = plt.plot(x, la[t][x]/255,linewidth=1,label='GroundTruth')
B, = plt.plot(x, lm1[t-512][x]/255,linewidth=1,label='Predicted',color ="red")
C, = plt.plot(x, lm2[t-512][x]/255,linewidth=1,label='Original',color ="green")

plt.xticks([])
plt.yticks(fontproperties = 'Times New Roman', size = 17)

# 设置图例并且设置图例的字体及大小
font1 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 17,
         }
legend = plt.legend(handles=[A, B, C], prop=font1,loc='lower right', numpoints = 1)
# legend = plt.legend(handles=[A, B, C], prop=font1,loc='upper left', numpoints = 1)
# legend = plt.legend(handles=[A, B, C], prop=font1,loc='lower left', numpoints = 1)
# 显示图像
plt.show()
