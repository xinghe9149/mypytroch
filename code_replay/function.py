import numpy as np
import pandas as pd
from PIL import Image
import math
def nearest(img, scale):
    '''
    该代码实现了最邻近插值算法，将图像放大到指定倍数
    参数：
    img: 输入图像
    scale: 放大倍数
    最后使用Image.fromarray()函数将结果转换为Image对象并输出
    '''
    # 获取原始图像的宽度和高度
    width, height, _ = img.shape
    # 计算新图像的宽度和高度
    new_width = width * scale
    new_height = height * scale
    # 创建一个具有相同颜色通道数的新图像数组，初始值为零
    new_img = np.zeros((new_width, new_height, 3))  # 3 for RGB
    # 遍历每个颜色通道
    for k in range(3):
        # 遍历新图像的每个像素
        for i in range(new_width):
            for j in range(new_height):
                # 将新图像的像素值映射到原始图像的最邻近像素值
                new_img[i, j, k] = img[round((i - 1) / scale), round((j - 1) / scale), k]
    # 将结果转换为Image对象并返回
    return Image.fromarray(np.uint8(new_img))

def double_linear(img,scale):
    '''
    该函数实现了双线性插值算法，将图像放大到指定倍数
    主要是通过输入图像中相邻的像素值来推算缩放（放大）后的图像像素的值
    '''
    width,height,_=img.shape
    new_width = int(width*scale)
    new_height = int(height*scale)
    new_img = np.zeros((new_width,new_height,3)) #3 for RGB
    for k in range(3):
        for i in range(new_width):
            for j in range(new_height):
                src_x = i/scale
                src_y = j/scale
                src_x_0 =int(np.floor(src_x))
                src_y_0 =int(np.floor(src_y))
                src_x_1 = min(src_x_0+1,width -1)
                src_y_1 = min(src_y_0+1,height -1)
                value0 = (src_x_1 - src_x)*img[src_x_0,src_y_0,k] + (src_x - src_x_0)*img[src_x_1,src_y_0,k]
                value1 =(src_x_1 - src_x)*img[src_x_0,src_y_1,k] + (src_x - src_x_0)*img[src_x_1,src_y_1,k]
                new_img[i,j,k] = int((src_y_1 - src_y)*value0 + (src_y - src_y_0)*value1)
    return Image.fromarray(np.uint8(new_img))

def bicubic_weight(x, a=-0.5):
    abs_x = abs(x)
    if abs_x <= 1:
        return (a + 2) * abs_x**3 - (a + 3) * abs_x**2 + 1
    elif 1 < abs_x < 2:
        return a * abs_x**3 - 5 * a * abs_x**2 + 8 * a * abs_x - 4 * a
    else:
        return 0


def bicubic(img,scale):
    '''
    该函数实现了双三次插值算法，将图像放大到指定倍数
    遍历周围16个像素值，通过双三次权重函数进行计算。
    边界处理：通过 if x+ii>=0 and x+ii<width and y+jj>=0 and y+jj<height:判断是否越界。
    使用np.clip()保证像素值在0-255之间。
    '''
    width,height,_=img.shape
    new_width = int(width*scale)
    new_height = int(height*scale)
    new_img = np.zeros((new_width,new_height,3)) #3 for RGB
    for k in range(3):
        for i in range(new_width):
            for j in range(new_height):
                src_x = i/scale
                src_y = j/scale
                x = math.floor(src_x)
                y = math.floor(src_y)
                x = int(x)
                y = int(y)
                u = src_x - x
                v = src_y - y
                pix = 0
                for ii in range(-1,3):
                    for jj in range(-1,3):
                        if x+ii>=0 and x+ii<width and y+jj>=0 and y+jj<height:
                            pix += img[x+ii,y+jj,k]*bicubic_weight(ii - u,-0.5)*bicubic_weight(jj - v,-0.5)
                new_img[i,j,k] = np.clip(pix,0,255)
    return Image.fromarray(np.uint8(new_img))

def nearest(img, scale):
    '''
    该代码实现了最邻近插值算法，将图像放大到指定倍数
    参数：
    img: 输入图像
    scale: 放大倍数
    最后使用Image.fromarray()函数将结果转换为Image对象并输出
    '''
    # 获取原始图像的宽度和高度
    width, height, _ = img.shape
    # 计算新图像的宽度和高度
    new_width = width * scale
    new_height = height * scale
    # 创建一个具有相同颜色通道数的新图像数组，初始值为零
    new_img = np.zeros((new_width, new_height, 3))  # 3 for RGB
    # 遍历每个颜色通道
    for k in range(3):
        # 遍历新图像的每个像素
        for i in range(new_width):
            for j in range(new_height):
                # 将新图像的像素值映射到原始图像的最邻近像素值
                new_img[i, j, k] = img[round((i - 1) / scale), round((j - 1) / scale), k]
    # 将结果转换为Image对象并返回
    return Image.fromarray(np.uint8(new_img))
def nearest(img, scale):
    '''
    该代码实现了最邻近插值算法，将图像放大到指定倍数
    参数：
    img: 输入图像
    scale: 放大倍数
    最后使用Image.fromarray()函数将结果转换为Image对象并输出
    '''
    # 获取原始图像的宽度和高度
    width, height, _ = img.shape
    # 计算新图像的宽度和高度
    new_width = width * scale
    new_height = height * scale
    # 创建一个具有相同颜色通道数的新图像数组，初始值为零
    new_img = np.zeros((new_width, new_height, 3))  # 3 for RGB
    # 遍历每个颜色通道
    for k in range(3):
        # 遍历新图像的每个像素
        for i in range(new_width):
            for j in range(new_height):
                # 将新图像的像素值映射到原始图像的最邻近像素值
                new_img[i, j, k] = img[round((i - 1) / scale), round((j - 1) / scale), k]
    # 将结果转换为Image对象并返回
    return Image.fromarray(np.uint8(new_img))