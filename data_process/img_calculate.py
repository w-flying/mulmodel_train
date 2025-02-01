import cv2
import numpy as np
from brisque import BRISQUE
import os
import pandas as pd
def calculate_colorfulness(image):
    # 将图像从BGR颜色空间转换为RGB颜色空间
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 分离RGB通道
    (R, G, B) = cv2.split(image.astype("float"))

    # 计算rg和yb通道
    rg = np.absolute(R - G)
    yb = np.absolute(0.5 * (R + G) - B)

    # 计算rg和yb通道的均值和标准差
    (rbMean, rbStd) = (np.mean(rg), np.std(rg))
    (ybMean, ybStd) = (np.mean(yb), np.std(yb))

    # 计算色彩丰富度
    stdRoot = np.sqrt((rbStd ** 2) + (ybStd ** 2))
    meanRoot = np.sqrt((rbMean ** 2) + (ybMean ** 2))
    colorfulness = stdRoot + (0.3 * meanRoot)
    return colorfulness

def calculate_brightness(image):
    # 将图像从BGR颜色空间转换为RGB颜色空间
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # 分离RGB通道
    (R, G, B) = cv2.split(image.astype("float"))
    # 根据公式计算亮度
    brightness = np.sqrt(0.241 * (R ** 2) + 0.391 * (G ** 2) + 0.068 * (B ** 2))
    # 计算整个图像的平均亮度
    average_brightness = np.mean(brightness)
    return average_brightness


def calculate_image_quality(image):
    # 这里假设image是RGB格式的彩色图像
    brisque_obj = BRISQUE()
    # 直接传入彩色图像
    score = brisque_obj.score(image)
    return score


def evaluate_contrast(image):
    # 转换为灰度图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 计算对比度
    contrast = np.std(gray)
    return contrast


img_files = os.listdir("../images")
cbq=[]
for img_file in img_files:
    image_path = os.path.join("../images", img_file)
    image = cv2.imread(image_path)
    try:
        colorfulness = calculate_colorfulness(image)
        brightness = calculate_brightness(image)
        quality = calculate_image_quality(image)
        contrast = evaluate_contrast(image)
        cbq.append([img_file, colorfulness, brightness, quality, contrast])
    except:
        cbq.append([img_file, -1, -1, -1])
pd.DataFrame(cbq, columns=['image', 'colorfulness', 'brightness', 'quality','contrast']).to_excel('Picture_features.xlsx', index=False)