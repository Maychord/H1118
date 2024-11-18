import numpy as np
import os

# 定义结果文件夹路径和 Ground Truth 文件列表
result_folder = "./Dataset0/result"
ground_truth_files = [
    "./Dataset0/Dataset0GroundTruth1.txt",
    "./Dataset0/Dataset0GroundTruth2.txt",
    "./Dataset0/Dataset0GroundTruth3.txt"
]

# 函数：计算两点之间的欧几里得距离
def calculate_distance(coord1, coord2):
    return np.sqrt((coord1[0] - coord2[0]) ** 2 + (coord1[1] - coord2[1]) ** 2)

# 初始化误差列表
errors = []

# 遍历每一个 Ground Truth 文件和对应的输出文件
for idx, gt_file in enumerate(ground_truth_files):
    output_file_path = f"{result_folder}/Dataset0OutputPos{idx+1}.txt"  # 假设输出文件为OutputPos1、OutputPos2、OutputPos3等

    # 读取 Ground Truth 和 Output 文件
    ground_truth = np.loadtxt(gt_file)
    output_data = np.loadtxt(output_file_path)

    # 检查两个文件是否匹配
    if ground_truth.shape[0] != output_data.shape[0]:
        print(f"Warning: {gt_file} 和 {output_file_path} 行数不匹配！")
        continue

    # 计算误差
    distances = [calculate_distance(ground_truth[i], output_data[i]) for i in range(len(ground_truth))]
    average_error = np.mean(distances)
    
    print(f"Dataset {idx + 1} 的平均误差: {average_error:.4f}")
    errors.append(average_error)

# 计算所有数据集的总体平均误差
overall_average_error = np.mean(errors)
print("所有数据集的总体平均误差:", overall_average_error)
