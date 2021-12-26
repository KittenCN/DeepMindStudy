import cv2
import os
from PIL import Image

img_size = 512

# 数据集来源
img_path = "emoji\data\train"

for path, dirs, files in os.walk(img_path, topdown=False):
    file_list = list(files)
for file in file_list:
    image_path = img_path + file
    img = cv2.imread(image_path, 1)
    bias = (img.shape[1] - img.shape[0]) // 2
    img = img[:, bias:bias+img.shape[0], :]
    (B, G, R) = cv2.split(img)
    # 颜色通道合并
    img = cv2.merge([R, G, B])
    # 缩放
    img = Image.fromarray(img)
    img = img.resize((img_size, img_size), Image.ANTIALIAS)
    os.remove(image_path)
    img.save(image_path.rstrip('.jpg') + '.png')
