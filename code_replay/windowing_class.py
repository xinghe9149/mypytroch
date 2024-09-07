import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt

class StatisticalNormalization(object):
    """
    Normalize an image by mapping intensity with intensity distribution
    """

    def __init__(self, sigma):
        self.name = 'StatisticalNormalization'
        assert isinstance(sigma, float)
        self.sigma = sigma

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        # 计算图像的统计值，如平均值和标准差
        statisticsFilter = sitk.StatisticsImageFilter()
        statisticsFilter.Execute(image)

        # 设置强度窗口化的最大和最小值
        intensityWindowingFilter = sitk.IntensityWindowingImageFilter()
        intensityWindowingFilter.SetOutputMaximum(255)  # 最大值映射为255
        intensityWindowingFilter.SetOutputMinimum(0)    # 最小值映射为0

        # 通过均值和sigma设置窗口的最大和最小值
        window_max = statisticsFilter.GetMean() + self.sigma * statisticsFilter.GetSigma()
        window_min = statisticsFilter.GetMean() - self.sigma * statisticsFilter.GetSigma()
        
        intensityWindowingFilter.SetWindowMaximum(window_max)
        intensityWindowingFilter.SetWindowMinimum(window_min)

        # 对图像应用窗口化处理
        image = intensityWindowingFilter.Execute(image)

        return {'image': image, 'label': label}

if __name__ == '__main__':
# 示例数据处理部分
    def load_image(path):
        """
        加载医学图像，例如CT图像
        """
        return sitk.ReadImage(path)

    def show_image(image, title=''):
        """
        显示医学图像
        """
        plt.figure(figsize=(6,6))
        plt.title(title)
        plt.imshow(sitk.GetArrayViewFromImage(image), cmap='gray')
        plt.axis('off')
        plt.show()


    # 加载CT图像
    image_path = 'your_ct_image_path.nii.gz'  # 替换为实际图像路径
    label_path = 'your_ct_label_path.nii.gz'  # 替换为实际标签路径

    ct_image = load_image(image_path)
    ct_label = load_image(label_path)

    # 创建示例
    sample = {'image': ct_image, 'label': ct_label}

    # 实例化正则化类，设定sigma参数（控制windowing窗口大小）
    sigma_value = 2.0  # 调整 sigma 值来更改窗口宽度
    normalizer = StatisticalNormalization(sigma=sigma_value)

    # 应用统计正则化和windowing
    processed_sample = normalizer(sample)

    # 显示原始图像和处理后的图像
    show_image(sample['image'], title="Original Image")
    show_image(processed_sample['image'], title="Processed Image with Windowing")
