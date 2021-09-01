import glob
import os
from PIL import Image

WIDTH = 544       # 图像的宽度
HEIGHT = 384      # 图像的高度


save_path = "./Images/TrainImage/train/"                   # 保存数据集的路径
pathTrain = "./oriange_images/Train_image/"                # 训练集图片路径
pathTruth = "./oriange_images/Lable_image/"                # 训练集标签路径


folder = os.path.exists(save_path)
if not folder:                          #判断是否存在文件夹如果不存在则创建为文件夹
	os.makedirs(save_path)


i = 1
	
for trainimg in glob.glob(os.path.join(pathTrain,"*.jpg")):
  imagefile = []  	
  img_name = os.path.basename(trainimg)
  lab_name = img_name[:-4]+".png"
  imagefile.append(Image.open(pathTrain + img_name))  # 读取训练集中的图片
  imagefile.append(Image.open(pathTruth + lab_name))  # 读取标签集中的图片


  target = Image.new('RGB', (WIDTH*2, HEIGHT))     # 创建一个空的图像width , height

  left = 0
  

  for image in imagefile:

    # target.paste(image)

    target.paste(image, (left, 0))
    left += WIDTH
    quality_value = 1000  # 图像品质
  target.save(save_path+'{}.jpg'.format(i), quality=quality_value)
  print("第%d张图片制作完成" % i)
  i+=1
