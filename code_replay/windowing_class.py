import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt

import SimpleITK as sitk

class StatisticalNormalization(object):
    """
    Normalize an image by mapping intensity with intensity distribution.
    Can be used for both 2D and 3D images.
    """

    def __init__(self, sigma):
        self.name = 'StatisticalNormalization'
        assert isinstance(sigma, float), "Sigma should be a float value."
        self.sigma = sigma

    def __call__(self, image):
        """
        对图像进行强度标准化。
        
        参数:
        - image (SimpleITK.Image): 输入的图像，可以是2D或3D。
        
        返回:
        - image (SimpleITK.Image): 标准化后的图像。
        """
        # 计算图像的统计值，如平均值和标准差
        statisticsFilter = sitk.StatisticsImageFilter()
        statisticsFilter.Execute(image)

        # 设置强度窗口化的最大和最小值
        intensityWindowingFilter = sitk.IntensityWindowingImageFilter()
        intensityWindowingFilter.SetOutputMaximum(255)  # 最大值映射为255
        intensityWindowingFilter.SetOutputMinimum(0)    # 最小值映射为0

        # 通过均值和 sigma 设置窗口的最大和最小值
        mean = statisticsFilter.GetMean()
        sigma = statisticsFilter.GetSigma()
        window_max = mean + self.sigma * sigma
        window_min = mean - self.sigma * sigma
        
        intensityWindowingFilter.SetWindowMaximum(window_max)
        intensityWindowingFilter.SetWindowMinimum(window_min)

        # 对图像应用窗口化处理
        normalized_image = intensityWindowingFilter.Execute(image)

        return normalized_image

if __name__ == '__main__':
    # 加载CT图像
    def load_image(path):
        return sitk.ReadImage(path)

    def show_images(original_image, processed_image, slice_index=None):
        """
        合并展示原始图像和处理后的图像
        """
        # 将SimpleITK对象转换为numpy数组
        original_array = sitk.GetArrayViewFromImage(original_image)
        processed_array = sitk.GetArrayViewFromImage(processed_image)
        
        # 如果没有指定切片索引，使用中间切片
        if slice_index is None:
            slice_index = original_array.shape[0] // 2
        
        # 获取指定的切片
        original_slice = original_array[slice_index]
        processed_slice = processed_array[slice_index]
        
        # 使用subplot将两张图片并排展示
        plt.figure(figsize=(12, 6))
        
        # 左图：原始图像
        plt.subplot(1, 2, 1)
        plt.title('Original Image')
        plt.imshow(original_slice, cmap='gray')
        plt.axis('off')
        
        # 右图：处理后的图像
        plt.subplot(1, 2, 2)
        plt.title('Processed Image with Windowing')
        plt.imshow(processed_slice, cmap='gray')
        plt.axis('off')
        
        # 显示图像
        plt.show()
    image_path = r'E:\Imageomics_dataset\Origin\images\2855.nii.gz'  # 替换为实际图像路径
    label_path = r'E:\Imageomics_dataset\Origin\masks\2855.nii.gz'  # 替换为实际标签路径

    ct_image = load_image(image_path)
    ct_label = load_image(label_path)

    # 创建示例
    sample = {'image': ct_image, 'label': ct_label}

    # 实例化正则化类，设定sigma参数（控制windowing窗口大小）
    sigma_value = 2.0  # 调整 sigma 值来更改窗口宽度
    normalizer = StatisticalNormalization(sigma=sigma_value)

    # 应用统计正则化和windowing
    processed_sample = normalizer(sample)

    # 合并显示原始图像和处理后的图像
    show_images(sample['image'], processed_sample['image'])
