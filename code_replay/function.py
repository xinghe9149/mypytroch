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


def normalize(slice, bottom=99, down=1):
    """  
    对区域非零值进行标准化，并将值裁剪到指定范围。
    
    该函数的主要目的是对图像的非背景部分（非零值）进行标准化，
    并通过裁剪掉顶部和底部的极端值来使数据集更加‘公平’。
    
    参数:
    - slice: 待处理的图像数据。
    - bottom: 底部百分位数，默认为99，用于确定裁剪的下限。
    - down: 顶部百分位数，默认为1，用于确定裁剪的上限。
    
    返回:
    - 标准化并裁剪后的图像数据。
    """
    # 确定裁剪的下限和上限
    b = np.percentile(slice, bottom)
    t = np.percentile(slice, down)
    # 裁剪图像数据，去除极端值
    slice = np.clip(slice, t, b)

    # 提取非零值
    image_nonzero = slice[np.nonzero(slice)]
    # 如果标准差为0，则直接返回原slice
    if np.std(slice) == 0 or np.std(image_nonzero) == 0:
        return slice
    else:
        # 对非零值进行标准化
        tmp = (slice - np.mean(image_nonzero)) / np.std(image_nonzero)
        # 将标准化后的最小值设为-9，以区分背景
        tmp[tmp == tmp.min()] = -9 
        return tmp
    
def crop_center(img, croph, cropw):
    """
    从图片中心裁剪出指定大小的区域。

    参数:
    img: 待裁剪的图片，假设图片序列的形状为 (N, H, W)，其中 N 是图片数量，H 是高度，W 是宽度。
    croph: 裁剪区域的高度。
    cropw: 裁剪区域的宽度。

    返回值:
    裁剪后的图片序列，裁剪区域从每张图片的中心开始计算。
    """
    # 初始化图片的高度和宽度
    height, width = img[0].shape
    # 计算裁剪区域的起始高度
    starth = height // 2 - (croph // 2)
    # 计算裁剪区域的起始宽度
    startw = width // 2 - (cropw // 2)
    # 返回裁剪后的图片序列
    return img[:, starth:starth + croph, startw:startw + cropw]
