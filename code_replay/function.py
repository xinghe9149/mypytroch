import numpy as np
import pandas as pd
from PIL import Image
def nearest(img,scale):
    '''
    该代码实现了最邻近插值算法，将图像放大到指定倍数
    参数：
    img:输入图像
    scale:放大倍数
    最后使用Image.fromarray()函数将结果转换为Image对象并输出
    '''
    width,height,_=img.shape
    new_width = width*scale
    new_height = height*scale
    new_img = np.zeros((new_width,new_height,3)) #3 for RGB
    for k in range(3):
        for i in range(new_width):
            for j in range(new_height):
                new_img[i,j,k] = img[round((i-1)/scale),round((j-1)/scale),k] #映射
    return Image.fromarray(np.uint8(new_img))

def double_linear(img,scale):
    '''
    该函数实现了双线性插值算法，将图像放大到指定倍数
    主要是通过输入图像中相邻的像素值来推算缩放（放大）后的图像像素的值
    '''
    width,height,_=img.shape
    new_width = width*scale
    new_height = height*scale
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
