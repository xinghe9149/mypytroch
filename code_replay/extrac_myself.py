import os
import pandas as pd
import radiomics
from radiomics import featureextractor  # This module provides the feature extraction functionality

# 提取特定特征并保存到CSV的函数
def extract_features_with_label(image_dir, mask_dir, output_csv, label=1):
    # 初始化特征抽取器，启用或禁用你感兴趣的特征
    extractor = featureextractor.RadiomicsFeatureExtractor()

    # 启用你需要的特征类别
    extractor.enableFeatureClassByName('firstorder')  # 一阶统计特征
    extractor.enableFeatureClassByName('glcm')        # 灰度共生矩阵特征

    # 数据存储的字典
    features_list = []

    # 遍历影像文件夹
    for image_file in os.listdir(image_dir):
        if image_file.endswith('.nii.gz'):
            # 获取影像文件路径和对应的掩码文件路径
            image_path = os.path.join(image_dir, image_file)
            mask_path = os.path.join(mask_dir, image_file.replace('image', 'mask'))  # 假设mask文件名与image对应

            if os.path.exists(mask_path):  # 检查掩码文件是否存在
                print(f"正在处理影像: {image_file} 和 掩码: {os.path.basename(mask_path)}")

                # 执行特征提取，指定标签
                try:
                    features = extractor.execute(image_path, mask_path, label=label)
                    feature_dict = {k: v for k, v in features.items()}  # 将特征存入字典
                    feature_dict['Image'] = image_file  # 添加影像名称用于标识
                    
                    features_list.append(feature_dict)
                except Exception as e:
                    print(f"处理 {image_file} 时出错: {e}")

    # 将提取到的所有特征存入DataFrame
    df = pd.DataFrame(features_list)

    # 删除包含 'diagnostics' 的列
    df = df.loc[:, ~df.columns.str.contains('diagnostics')]

    # 保存到CSV文件
    df.to_csv(output_csv, index=False)
    print(f"特征提取完成，已保存至 {output_csv}")

# 影像和掩码文件夹路径
image_dir = r'E:\Imageomics_dataset\Origin\images'  # 替换为你的影像文件夹路径
mask_dir = r'E:\Imageomics_dataset\Origin\masks'  # 替换为你的掩码文件夹路径
output_csv = 'extracted_features.csv'  # 输出CSV文件

# 执行特征提取，指定感兴趣的标签 (例如标签值为1)
extract_features_with_label(image_dir, mask_dir, output_csv, label=1)
