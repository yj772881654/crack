from utils.utils import *
from utils.Validator import *
from utils.Crackloader import *
import os
netName = "CANet_New"
valid_log_dir = "./log/" + netName
best_model_dir = "./model/" + netName + "/"
image_format = "jpg"
lable_format = "bmp"
datasetName = "CrackLS315"
# datasetName="CrackTree"
valid_img_dir = "./datasets/" + datasetName + "/valid/Valid_image/"
valid_lab_dir = "./datasets/" + datasetName + "/valid/Lable_image/"
# pretrain_dir="/home/nlg/yj/mxy/code/crack/model/CrackTree/0.570.pkl"
pretrain_dir="model/0.820.pkl"
valid_result_dir = "./datasets/" + datasetName + "/valid/Valid_result/"
def Test():
    crack = torch.load(pretrain_dir).cuda().eval()
    validator = Validator(valid_img_dir, valid_lab_dir,
                          valid_result_dir, valid_log_dir, best_model_dir, crack, image_format, lable_format)
    validator.validate(0)

if __name__ == '__main__':
    Test()