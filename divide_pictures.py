
import os
import shutil
import random
from random import shuffle


def split_dataset(input_folder, output_folder, train_ratio=0.7, seed=None):

    """
    将照片文件夹按照指定的比例划分为训练集和验证集

    参数：
    - input_folder: 输入照片文件夹路径
    - output_folder: 输出文件夹路径，包含train和val子文件夹
    - train_ratio: 训练集比例，默认为0.7
    - seed: 随机种子，可选参数

    返回值：无
    """

    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)
    train_folder = os.path.join(output_folder, 'train')
    val_folder = os.path.join(output_folder, 'val')

    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(val_folder, exist_ok=True)

    # 获取照片文件列表
    photo_files = [f for f in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, f))]
    random.shuffle(photo_files)

    # 计算划分数量
    total_count = len(photo_files)
    train_count = int(total_count * train_ratio)

    # 划分并复制文件
    train_files = photo_files[:train_count]
    val_files = photo_files[train_count:]

    for file in train_files:
        shutil.copy(os.path.join(input_folder, file), os.path.join(train_folder, file))

    for file in val_files:
        shutil.copy(os.path.join(input_folder, file), os.path.join(val_folder, file))

# 用法示例


input_folder_path = r'D:\DEEPLEARNING\Deeplearning\Bijie_landslide_dataset\non_landslidedevide/'
output_folder_path = r'D:\DEEPLEARNING\Deeplearning\Bijie_landslide_dataset\2_non/'
split_dataset(input_folder_path, output_folder_path)
