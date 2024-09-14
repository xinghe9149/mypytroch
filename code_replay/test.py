from loading_data import MedicalImageLoader
from function_class import *
import os
import numpy as np
import nibabel as nib
# 目录
class MedicalImageBatchProcessor:
    """
    批量处理医学图像的类。支持2D和3D图像的加载、处理（标准化、裁剪）和保存。
    """
    
    def __init__(self, image_directory, save_directory):
        """
        初始化类，设置图像的加载目录和保存目录。
        
        参数:
        - image_directory (str): 存放图像的文件夹路径。
        - save_directory (str): 保存处理后图像的文件夹路径。
        """
        self.image_directory = image_directory
        self.save_directory = save_directory
        
        # 确保保存目录存在
        if not os.path.exists(self.save_directory):
            os.makedirs(self.save_directory)
    
    def save_image_2d(self, image_data, save_path):
        """
        保存2D图像为PNG格式。
        
        参数:
        - image_data (numpy.ndarray): 处理后的2D图像数据。
        - save_path (str): 保存路径（包含文件名和扩展名）。
        """
        img = Image.fromarray(np.uint8(image_data))
        img.save(save_path)
        print(f"Saved 2D image to {save_path}")

    def save_image_3d(self, image_data, save_path):
        """
        保存3D图像为NIfTI格式。
        
        参数:
        - image_data (numpy.ndarray): 处理后的3D图像数据。
        - save_path (str): 保存路径（包含文件名和扩展名）。
        """
        img_nii = nib.Nifti1Image(image_data, affine=np.eye(4))  # 使用单位矩阵作为仿射变换矩阵
        nib.save(img_nii, save_path)
        print(f"Saved 3D image to {save_path}")
    
    def process_images(self):
        """
        遍历指定目录中的图像文件，逐个加载、处理并保存结果。
        """
        for filename in os.listdir(self.image_directory):
            file_path = os.path.join(self.image_directory, filename)
            
            # 打印当前处理的文件名
            print(f"Processing file: {filename}")
            
            # 加载图像
            try:
                loader = MedicalImageLoader(file_path)
                image_data = loader.load_image()
            except Exception as e:
                print(f"Failed to load {filename}: {e}")
                continue

            # 根据图像的维度选择处理器
            if image_data.ndim == 2:  # 2D 图像
                print(f"Processing 2D image: {filename}")
                processor = MedicalImageProcessor2D(image_data)

                # 执行标准化处理
                normalized_image = processor.normalize()
                # 裁剪图像中心，假设裁剪到 256x256
                cropped_image = processor.crop_center(256, 256)
                #最邻近插值算法
                #nearest_image = processor.nearest(0.5)
                #双线性插值算法
                #double_linear_image = processor.double_linear( 0.5)
                #双三次插值算法
                #bicubic_image = processor.bicubic(0.5)

                # 保存裁剪后的图像
                save_path = os.path.join(self.save_directory, f"processed_{filename}")
                self.save_image_2d(cropped_image, save_path)

            elif image_data.ndim == 3:  # 3D 图像
                print(f"Processing 3D image: {filename}")
                processor = MedicalImageProcessor3D(image_data)

                # 执行标准化和裁剪处理
                normalized_image = processor.normalize_and_clip()

                # 裁剪图像中心，假设裁剪到 128x128x128
                cropped_image = processor.crop_center(60, 128, 128)
                result_image = processor.apply_statistical_normalization(0.5)
                # 保存裁剪后的图像
                save_path = os.path.join(self.save_directory, f"processed_{os.path.splitext(filename)[0]}.gz")
                self.save_image_3d(result_image, save_path)

            else:
                print(f"Unsupported image dimensions for {filename}: {image_data.shape}")
# 设置要处理的图像目录和保存目录
image_directory = r"E:\Imageomics_dataset\Origin\images"
save_directory = r"E:\Imageomics_dataset\output_myself\images"

# 创建批量处理器实例
processor = MedicalImageBatchProcessor(image_directory, save_directory)

# 开始处理图像
processor.process_images()
