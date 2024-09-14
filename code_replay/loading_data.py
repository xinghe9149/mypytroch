import numpy as np
import nibabel as nib
import SimpleITK as sitk
from PIL import Image
import os

class MedicalImageLoader:
    """
    用于导入 2D 和 3D 医学图像的类。支持 .nii/.nii.gz 格式的 3D 图像，
    以及常见的 2D 图像格式（如 .png, .jpg, .bmp）。
    """
    
    def __init__(self, file_path):
        """
        初始化类，接受输入图像的路径。
        
        参数:
        - file_path (str): 图像文件路径。
        """
        self.file_path = file_path  # 图像路径
        self.image_data = None      # 存储导入的图像数据
    
    def load_image(self):
        """
        导入图像，根据图像的类型（2D 或 3D）执行不同的加载操作。
        
        返回:
        - image_data (numpy.ndarray): 加载后的图像数据。
        """
        # 检查文件扩展名以确定是 2D 还是 3D 图像
        file_ext = os.path.splitext(self.file_path)[1].lower()

        # 如果是 NIfTI 格式，加载 3D 图像
        if file_ext in ['.nii', '.gz']:
            self.image_data = self._load_3d_image()
        
        # 否则假设是 2D 图像（支持常见的 2D 图像格式）
        elif file_ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']:
            self.image_data = self._load_2d_image()
        
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")
        
        return self.image_data
    
    def _load_3d_image(self):
        """
        加载 NIfTI 格式的 3D 医学图像。
        
        返回:
        - img_data (numpy.ndarray): 加载后的 3D 图像数据。
        """
        # 使用 nibabel 读取 NIfTI 图像
        img_nii = nib.load(self.file_path)
        img_data = img_nii.get_fdata()  # 获取 3D 图像数据
        print(f"Loaded 3D image with shape: {img_data.shape}")
        
        return img_data
    
    def _load_2d_image(self):
        """
        加载常见格式的 2D 图像。
        
        返回:
        - img_data (numpy.ndarray): 加载后的 2D 图像数据。
        """
        # 使用 PIL 读取 2D 图像
        img = Image.open(self.file_path).convert('L')  # 将图像转换为灰度模式 (L)
        img_data = np.array(img)  # 转换为 NumPy 数组
        print(f"Loaded 2D image with shape: {img_data.shape}")
        
        return img_data
    
    def get_image(self):
        """
        返回加载的图像数据。
        
        返回:
        - image_data (numpy.ndarray): 图像数据。
        """
        if self.image_data is not None:
            return self.image_data
        else:
            raise ValueError("No image has been loaded. Please call 'load_image' first.")
if __name__ == "__main__":
    # 示例1: 加载 2D 图像
    file_path_2d = "example_2d.png"
    loader_2d = MedicalImageLoader(file_path_2d)
    img_data_2d = loader_2d.load_image()
    print(f"2D image shape: {img_data_2d.shape}")

    # 示例2: 加载 3D 图像
    file_path_3d = "example_3d.nii.gz"
    loader_3d = MedicalImageLoader(file_path_3d)
    img_data_3d = loader_3d.load_image()
    print(f"3D image shape: {img_data_3d.shape}")
