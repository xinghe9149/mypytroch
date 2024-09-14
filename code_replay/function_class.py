import numpy as np
import math
from PIL import Image

class MedicalImageProcessor2D:
    """
    处理2D医学图像的类，包括中心裁剪和标准化处理。
    """

    def __init__(self, image):
        """
        初始化类，接受输入图像并检查维度。
        
        参数:
        - image (numpy.ndarray): 输入的2D图像数据。
        """
        self.image = image
        self._check_dimensions()

    def _check_dimensions(self):
        """
        检查图像的维度是否为2D。如果不是，抛出错误。
        """
        if self.image.ndim != 2:
            raise ValueError("Error: The input image is not 2D. Please provide a 2D image.")

    def normalize(self, bottom=99, down=1):
        """
        对图像进行标准化并裁剪极值。
        
        参数:
        - bottom (float): 高值裁剪百分比。
        - down (float): 低值裁剪百分比。
        
        返回:
        - 经过标准化处理的图像。
        """
        b = np.percentile(self.image, bottom)
        t = np.percentile(self.image, down)
        normalized_image = np.clip(self.image, t, b)
        image_nonzero = normalized_image[np.nonzero(normalized_image)]

        if np.std(normalized_image) == 0 or np.std(image_nonzero) == 0:
            return normalized_image
        else:
            tmp = (normalized_image - np.mean(image_nonzero)) / np.std(image_nonzero)
            tmp[tmp == tmp.min()] = -9
            return tmp

    def crop_center(self, croph, cropw):
        """
        从图像中心裁剪出指定大小的区域。
        
        参数:
        - croph (int): 裁剪区域的高度。
        - cropw (int): 裁剪区域的宽度。
        
        返回:
        - 裁剪后的图像。
        """
        height, width = self.image.shape
        starth = height // 2 - (croph // 2)
        startw = width // 2 - (cropw // 2)
        return self.image[starth:starth + croph, startw:startw + cropw]
    def nearest(self, scale):
        '''
        实现最邻近插值算法，将图像放大到指定倍数。
        
        参数:
        - scale: 放大倍数
        
        返回:
        - 处理后的图像
        '''
        width, height, _ = self.image.shape
        new_width = width * scale
        new_height = height * scale
        new_img = np.zeros((new_width, new_height, 3))  # 3 for RGB
        for k in range(3):
            for i in range(new_width):
                for j in range(new_height):
                    new_img[i, j, k] = self.image[round((i - 1) / scale), round((j - 1) / scale), k]
        return Image.fromarray(np.uint8(new_img))

    def double_linear(self, scale):
        '''
        实现双线性插值算法，将图像放大到指定倍数。
        
        参数:
        - scale: 放大倍数
        
        返回:
        - 处理后的图像
        '''
        width, height, _ = self.image.shape
        new_width = int(width * scale)
        new_height = int(height * scale)
        new_img = np.zeros((new_width, new_height, 3))  # 3 for RGB
        for k in range(3):
            for i in range(new_width):
                for j in range(new_height):
                    src_x = i / scale
                    src_y = j / scale
                    src_x_0 = int(np.floor(src_x))
                    src_y_0 = int(np.floor(src_y))
                    src_x_1 = min(src_x_0 + 1, width - 1)
                    src_y_1 = min(src_y_0 + 1, height - 1)
                    value0 = (src_x_1 - src_x) * self.image[src_x_0, src_y_0, k] + (src_x - src_x_0) * self.image[src_x_1, src_y_0, k]
                    value1 = (src_x_1 - src_x) * self.image[src_x_0, src_y_1, k] + (src_x - src_x_0) * self.image[src_x_1, src_y_1, k]
                    new_img[i, j, k] = int((src_y_1 - src_y) * value0 + (src_y - src_y_0) * value1)
        return Image.fromarray(np.uint8(new_img))

    def bicubic(self, scale):
        '''
        实现双三次插值算法，将图像放大到指定倍数。
        
        参数:
        - scale: 放大倍数
        
        返回:
        - 处理后的图像
        '''
        def bicubic_weight(x, a=-0.5):
            abs_x = abs(x)
            if abs_x <= 1:
                return (a + 2) * (abs_x ** 3) - (a + 3) * (abs_x ** 2) + 1
            elif 1 < abs_x < 2:
                return a * (abs_x ** 3) - 5 * a * (abs_x ** 2) + 8 * a * abs_x - 4 * a
            else:
                return 0

        width, height, _ = self.image.shape
        new_width = int(width * scale)
        new_height = int(height * scale)
        new_img = np.zeros((new_width, new_height, 3))  # 3 for RGB
        for k in range(3):
            for i in range(new_width):
                for j in range(new_height):
                    src_x = i / scale
                    src_y = j / scale
                    x = math.floor(src_x)
                    y = math.floor(src_y)
                    u = src_x - x
                    v = src_y - y
                    pix = 0
                    for ii in range(-1, 3):
                        for jj in range(-1, 3):
                            if 0 <= x + ii < width and 0 <= y + jj < height:
                                pix += self.image[x + ii, y + jj, k] * bicubic_weight(ii - u) * bicubic_weight(jj - v)
                    new_img[i, j, k] = np.clip(pix, 0, 255)
        return Image.fromarray(np.uint8(new_img))

import numpy as np
import SimpleITK as sitk
import scipy.ndimage
from windowing_class import StatisticalNormalization
class MedicalImageProcessor3D:
    """
    用于处理 3D 医学图像的类，包含缩放、中心裁剪和标准化功能。
    """

    def __init__(self, img_data):
        """
        初始化类，接受 3D 图像数据。
        
        参数:
        - img_data (numpy.ndarray): 输入的 3D 图像数据。
        """
        self.img_data = img_data  # 3D 图像数据
        self._check_dimensions()
        self.original_shape = img_data.shape  # 图像原始形状
        
    def _check_dimensions(self):
        """
        检查图像的维度是否为3D。如果不是，抛出错误。
        """
        if self.img_data.ndim != 3:
            raise ValueError("Error: The input image is not 3D. Please provide a 3D image.")
    def resize(self, scale, order=3):
        """
        缩放 3D 图像。
        
        参数:
        - scale (float): 缩放倍数（大于1表示放大，小于1表示缩小）。
        - order (int): 插值顺序，默认为 3 (双三次插值)。
                       0: 最近邻插值
                       1: 双线性插值
                       3: 双三次插值
        
        返回:
        - zoomed_img (numpy.ndarray): 缩放后的 3D 图像数据。
        """
        print(f"Original shape: {self.original_shape}")

        # 计算新的尺寸
        new_shape = tuple([int(dim * scale) for dim in self.original_shape])
        print(f"New shape: {new_shape}")

        # 使用 scipy.ndimage.zoom 进行 3D 插值
        zoomed_img = scipy.ndimage.zoom(self.img_data, zoom=scale, order=order)
        
        return zoomed_img
    
    def crop_center(self, crop_depth, crop_height, crop_width):
        """
        从 3D 图像中心裁剪出指定大小的区域。
        
        参数:
        - crop_depth: 裁剪区域的深度。
        - crop_height: 裁剪区域的高度。
        - crop_width: 裁剪区域的宽度。
        
        返回:
        - cropped_img (numpy.ndarray): 中心裁剪后的 3D 图像数据。
        """
        depth, height, width = self.img_data.shape
        
        # 计算中心点坐标
        start_d = depth // 2 - crop_depth // 2
        start_h = height // 2 - crop_height // 2
        start_w = width // 2 - crop_width // 2
        
        # 裁剪出 3D 图像的中心区域
        cropped_img = self.img_data[start_d:start_d + crop_depth, 
                                    start_h:start_h + crop_height, 
                                    start_w:start_w + crop_width]
        
        print(f"Cropped image shape: {cropped_img.shape}")
        
        return cropped_img
    
    def normalize_and_clip(self, bottom=99, top=1):
        """
        对 3D 图像的非零区域进行标准化，并裁剪掉上下的极端值。
        
        参数:
        - bottom: 底部百分位数，默认为99，用于裁剪上限。
        - top: 顶部百分位数，默认为1，用于裁剪下限。
        
        返回:
        - normalized_img (numpy.ndarray): 标准化并裁剪后的 3D 图像数据。
        """
        # 计算指定百分位数的上下限
        lower_percentile = np.percentile(self.img_data, top)
        upper_percentile = np.percentile(self.img_data, bottom)
        
        # 对图像进行裁剪，限制数据在上下限范围内
        clipped_img = np.clip(self.img_data, lower_percentile, upper_percentile)
        
        # 提取非零部分进行标准化
        non_zero_voxels = clipped_img[clipped_img > 0]
        
        if non_zero_voxels.size > 0:
            mean_val = np.mean(non_zero_voxels)
            std_val = np.std(non_zero_voxels)
            
            if std_val != 0:
                normalized_img = (clipped_img - mean_val) / std_val
                normalized_img[clipped_img == 0] = -9  # 保留背景为特殊值 -9
            else:
                normalized_img = clipped_img
        else:
            normalized_img = clipped_img
        
        print("Normalization complete.")
        
        return normalized_img

    def transform_ctdata(self, windowWidth, windowLevel, normal=False):
        """
        根据窗口宽度和窗口水平对CT图像进行转换。
        
        参数:
        - windowWidth (float): 窗口宽度，决定了图像的对比度范围。
        - windowLevel (float): 窗口水平，决定了图像的亮度水平。
        - normal (bool): 是否将图像正规化到0-255的范围，默认为False。
        
        返回:
        - transformed_img (numpy.ndarray): 根据窗口水平和窗口宽度裁剪后的图像。
        
        注意:
        - 这个函数的self.img_data必须是float类型的，否则无效！
        """
        # 计算窗口的下界和上界
        minWindow = float(windowLevel) - 0.5 * float(windowWidth)
        maxWindow = float(windowLevel) + 0.5 * float(windowWidth)
        
        # 对图像进行线性转换，使其适应窗口宽度
        transformed_img = (self.img_data - minWindow) / float(windowWidth)
        
        # 将图像中小于0的值设为0，大于1的值设为1
        transformed_img[transformed_img < 0] = 0
        transformed_img[transformed_img > 1] = 1
        
        # 如果不进行正规化，则将图像转换到0-255的uint8范围
        if not normal:
            transformed_img = (transformed_img * 255).astype('uint8')
        
        print(f"CT image transformed with window level: {windowLevel} and window width: {windowWidth}")
        
        return transformed_img
    def apply_statistical_normalization(self, sigma):
        
        """
        使用 StatisticalNormalization 对 3D 图像进行强度标准化。
        
        参数:
        - sigma (float): 用于标准化的 sigma 值。
        
        返回:
        - normalized_img (SimpleITK.Image): 标准化后的 3D 图像。
        """
        if not isinstance(self.img_data, sitk.Image):
            sitk_image = self.numpy_to_sitk()
            #raise TypeError("The input image must be a SimpleITK Image for this operation.")
        
        # 调用 StatisticalNormalization 进行强度标准化
        normalization = StatisticalNormalization(sigma)
        normalized_img = normalization(sitk_image)
        normalized_img = self.sitk_to_numpy(normalized_img)
        print(f"Applied statistical normalization with sigma={sigma}.")
        return normalized_img
    def sitk_to_numpy(self, sitk_img):
        """
        将 SimpleITK.Image 转换回 numpy.ndarray 格式。
        
        参数:
        - sitk_img (SimpleITK.Image): SimpleITK 图像。
        
        返回:
        - numpy_img (numpy.ndarray): 转换后的 numpy 数组图像。
        """
        numpy_img = sitk.GetArrayFromImage(sitk_img)
        return numpy_img
    def numpy_to_sitk(self):
        """
        将 numpy.ndarray 格式的图像数据转换为 SimpleITK.Image。
        
        返回:
        - sitk_img (SimpleITK.Image): 转换后的 SimpleITK 图像。
        """
        sitk_img = sitk.GetImageFromArray(self.img_data)
        return sitk_img



# 示例用法
if __name__ == '__main__':

# 示例1：处理2D医学图像
    image_2d = np.random.rand(512, 512)  # 假设是加载的2D图像数据
    processor_2d = MedicalImageProcessor2D(image_2d)
    normalized_2d = processor_2d.normalize()
    cropped_2d = processor_2d.crop_center(256, 256)

    # 示例2：处理3D医学图像
    image_3d = np.random.rand(128, 128, 64)  # 假设是加载的3D图像数据
    processor_3d = MedicalImageProcessor3D(image_3d)
    resized_3d = processor_3d.resize(scale=1.5)
    normalized_3d = processor_3d.normalize()
    cropped_3d = processor_3d.crop_center(64, 64, 32)
