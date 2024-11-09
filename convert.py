import os
import numpy as np
from PIL import Image

# 获取当前目录
current_folder = os.getcwd()

# 遍历当前目录中的所有 PNG 文件
for file_name in os.listdir(current_folder):
    if file_name.endswith(".bmp"):
        # 构建文件路径
        img_path = os.path.join(current_folder, file_name)
        
        # 打开图像并将其转换为 numpy 数组
        img = Image.open(img_path).convert('RGB')
        img_array = np.array(img)
        
        # 构建 .npy 文件路径
        npy_file_name = os.path.splitext(file_name)[0] + ".npy"
        npy_path = os.path.join(current_folder, npy_file_name)
        
        # 将 numpy 数组保存为 .npy 文件
        np.save(npy_path, img_array)
        
        # 删除原始的 PNG 文件
        os.remove(img_path)
        print(f"已将 {img_path} 转换为 {npy_path} 并删除原始 PNG 文件")