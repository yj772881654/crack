import matplotlib as mpl
import numpy as np
mpl.use('Agg')
import cv2

imgName = "0023-1.jpg"
att_path = "./test5/"
img_path = att_path + imgName
attention1_path = att_path + imgName + "attention1.jpg"
attention2_path = att_path + imgName + "attention2.jpg"
attention3_path = att_path + imgName + "attention3.jpg"
attention4_path = att_path + imgName + "attention4.jpg"
attention5_path = att_path + imgName + "attention5.jpg"
# attention1_path = att_path + "epoch010_side2.png"
# attention2_path = att_path + "epoch010_side3.png"
# attention3_path = att_path + "epoch010_side4.png"
# attention4_path = att_path + "epoch010_side5.png"

# attention5_path = att_path + imgName +"./attention5.jpg"
img = cv2.imread(img_path)

heatmap1 = cv2.imread(attention1_path)
heatmap2 = cv2.imread(attention2_path)
heatmap3 = cv2.imread(attention3_path)
heatmap4 = cv2.imread(attention4_path)
heatmap5 = cv2.imread(attention5_path)

heatmap1 = cv2.applyColorMap(heatmap1, cv2.COLORMAP_JET)
heatmap2 = cv2.applyColorMap(heatmap2, cv2.COLORMAP_JET)
heatmap3 = cv2.applyColorMap(heatmap3, cv2.COLORMAP_JET)
heatmap4 = cv2.applyColorMap(heatmap4, cv2.COLORMAP_JET)
heatmap5 = cv2.applyColorMap(heatmap4, cv2.COLORMAP_JET)

superimposed_img1 = heatmap1*0.8+img*0.4
superimposed_img2 = heatmap2*0.8+img*0.4
superimposed_img3 = heatmap3*0.8+img*0.4
superimposed_img4 = heatmap4*0.8+img*0.4
superimposed_img5 = heatmap4*0.8+img*0.4

# print(img.shape)
# print(img.shape[1])
h = img.shape[0]
w = img.shape[1]
#
final_matrix = np.zeros((h, w*5, 3))
# change
final_matrix[0:h, 0:w] = superimposed_img3
final_matrix[0:h, w:w*2] = superimposed_img2
final_matrix[0:h, w*2:w*3] = superimposed_img1
final_matrix[0:h, w*3:w*4] = superimposed_img4
final_matrix[0:h, w*4:w*5] = superimposed_img5
# cv2.imwrite('./_img.jpg', superimposed_img3)
cv2.imwrite(att_path + './super2.jpg', final_matrix)