import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import zoom  # 直接使用 scipy 库来实现高效插值
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

    '''
    该函数实现三线性插值，用于将3D医学图像放大到指定倍数。
    参数：
    img: 输入的 3D 图像数组
    scale: 放大倍数
    返回放大后的图像数组
    '''
    depth, height, width = img.shape  # 获取原始图像的三维尺寸
    
    # 计算放大后的新尺寸
    new_depth = int(depth * scale)
    new_height = int(height * scale)
    new_width = int(width * scale)

    # 使用 scipy 的 zoom 函数进行三线性插值处理（order=1 表示线性插值）
    new_img = zoom(img, (scale, scale, scale), order=1)
    
    return new_img
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

def nearest_interpolation(img, scale):
    '''
    实现最近邻插值，将3D图像放大到指定倍数
    参数：
    img: 输入的 3D 图像数组
    scale: 放大倍数
    '''
    # 获取原始图像的宽度、高度和深度（假设是 3D 图像）
    depth, height, width = img.shape

    # 计算新的放大图像尺寸
    new_depth = depth * scale
    new_height = height * scale
    new_width = width * scale

    # 创建一个新的空数组来存储插值后的图像
    new_img = np.zeros((new_depth, new_height, new_width))

    # 遍历新图像的每个体素，并将其映射到原图像中的最邻近像素
    for i in range(new_depth):
        for j in range(new_height):
            for k in range(new_width):
                # 最近邻插值计算
                new_img[i, j, k] = img[round((i - 1) / scale), round((j - 1) / scale), round((k - 1) / scale)]
    
    return new_img

def trilinear_interpolation(img, scale):
    '''
    该函数实现三线性插值，用于将3D医学图像放大到指定倍数。
    参数：
    img: 输入的 3D 图像数组
    scale: 放大倍数
    返回放大后的图像数组
    '''
    depth, height, width = img.shape  # 获取原始图像的三维尺寸
    
    # 计算放大后的新尺寸
    new_depth = int(depth * scale)
    new_height = int(height * scale)
    new_width = int(width * scale)

    # 使用 scipy 的 zoom 函数进行三线性插值处理（order=1 表示线性插值）
    new_img = zoom(img, (scale, scale, scale), order=1)
    
    return new_img


import nibabel as nib
import numpy as np
import scipy.ndimage

def resize_3d_image(image_path, output_path, scale, order=3):
    """
    该函数对 3D 医学图像进行插值缩放，并保存结果。
    
    参数:
    image_path (str): 输入的 NIfTI 3D 图像路径 (.nii.gz)。
    output_path (str): 缩放后保存的 NIfTI 3D 图像路径。
    scale (float): 缩放倍数（大于1表示放大，小于1表示缩小）。
    order (int): 插值顺序，默认值为3表示双三次插值。
                 0: 最近邻插值
                 1: 双线性插值
                 3: 双三次插值 (默认)
    
    返回:
    None
    """
    # 读取 NIfTI 格式的 3D 图像
    img_nii = nib.load(image_path)
    img_data = img_nii.get_fdata()

    # 获取原始图像的形状 (宽, 高, 深度)
    original_shape = img_data.shape
    print(f"Original shape: {original_shape}")

    # 计算新的尺寸
    new_shape = tuple([int(dim * scale) for dim in original_shape])
    print(f"New shape: {new_shape}")

    # 使用 scipy 的 zoom 进行 3D 插值
    zoomed_img = scipy.ndimage.zoom(img_data, zoom=scale, order=order)

    # 创建新的 NIfTI 图像并保存
    zoomed_img_nii = nib.Nifti1Image(zoomed_img, img_nii.affine)
    return zoomed_img_nii


def transform_ctdata(self, windowWidth, windowLevel, normal=False):
    """
    根据窗口宽度和窗口水平对CT图像进行转换。
    
    参数:
    - windowWidth: 窗口宽度，决定了图像的对比度范围。
    - windowLevel: 窗口水平，决定了图像的亮度水平。
    - normal: 是否进行正规化到0-255的范围，默认为False。
    
    返回值:
    - 返回根据窗口水平和窗口宽度裁剪过的图像。
    
    注意:
    - 这个函数的self.image一定得是float类型的，否则就无效！
    """
    # 计算窗口的下界
    minWindow = float(windowLevel) - 0.5*float(windowWidth)
    
    # 对图像进行线性转换，使其适应窗口宽度
    newimg = (self.image - minWindow) / float(windowWidth)
    
    # 将图像中小于0的值设为0
    newimg[newimg < 0] = 0
    
    # 将图像中大于1的值设为1
    newimg[newimg > 1] = 1
    
    # 如果不进行正规化，则将图像转换到0-255的uint8范围
    if not normal:
        newimg = (newimg * 255).astype('uint8')
    
    # 返回转换后的图像
    return newimg