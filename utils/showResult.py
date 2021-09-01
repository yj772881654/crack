import glob
import os

import cv2
from PIL import Image



netName = "ResFusion2"
datasetName = "CFD"
best_thread = 0.35

save_path = "../results/" + netName + "/" + datasetName + "/"  # 保存数据集的路径
pred_path = "../datasets/" + datasetName + "/test/" + netName + "_test_res/"


folder = os.path.exists(save_path)
if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
    os.makedirs(save_path)

i = 1

for predimg in glob.glob(os.path.join(pred_path, "*.jpg")):
    imagefile = []
    img_name = os.path.basename(predimg)
    gray = cv2.imread(predimg)

    # lab_name = img_name[:-4] + ".png"

    ret, img = cv2.threshold(gray, 255 * best_thread, 255, cv2.THRESH_BINARY)
    cv2.imwrite(save_path + img_name, img)

    # target = Image.new('RGB', (WIDTH * 2, HEIGHT))  # 创建一个空的图像width , height

    # left = 0
    #
    # for image in imagefile:
    #     # target.paste(image)
    #
    #     target.paste(image, (left, 0))
    #     left += WIDTH
    #     quality_value = 1000  # 图像品质
    # target.save(save_path + '{}.jpg'.format(i), quality=quality_value)
    print("第%d张图片制作完成" % i)
    i += 1