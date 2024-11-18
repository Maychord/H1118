import numpy as np
import itertools

# func to read in slices
def read_slice_of_file(file_path, start, end):
    with open(file_path, 'r') as file:
        # use itertools.islice to get slices
        slice_lines = list(itertools.islice(file, start, end))
    return slice_lines

# 读取配置文件信息
def read_cfg_file(cfg_path):
    slice_lines = read_slice_of_file(cfg_path, 1, 6)
    info = np.loadtxt(slice_lines)
    tol_samp_num = int(info[0])
    port_num = int(info[2])
    ant_num = int(info[3])
    sc_num = int(info[4])
    return tol_samp_num, port_num, ant_num, sc_num

# 主程序入口
if __name__ == "__main__":
    # 定义配置和输入文件的路径格式
    base_path = '/Users/chunz/Downloads/hack/Dataset0/'
    cfg_files = [f'{base_path}Dataset0CfgData{i}.txt' for i in range(1, 4)]
    input_data_files = [f'{base_path}Dataset0InputData{i}.txt' for i in range(1, 4)]
    
    for i in range(3):  # 遍历三个数据集
        print(f"Processing Data {i+1}")
        
        cfg_path = cfg_files[i]
        inputdata_path = input_data_files[i]
        
        # 读取配置文件信息
        tol_samp_num, port_num, ant_num, sc_num = read_cfg_file(cfg_path)
        
        # 读取信道数据文件
        H = []
        slice_samp_num = 1000   # 每个分片的样本数
        slice_num = int(tol_samp_num / slice_samp_num)  # 分片总数
        
        for slice_idx in range(slice_num):
            print(f"Loading slice {slice_idx + 1}/{slice_num} for Data {i+1}")
            slice_lines = read_slice_of_file(inputdata_path, slice_idx * slice_samp_num, (slice_idx + 1) * slice_samp_num)
            Htmp = np.loadtxt(slice_lines)
            Htmp = np.reshape(Htmp, (slice_samp_num, 2, sc_num, ant_num, port_num))
            Htmp = Htmp[:, 0, :, :, :] + 1j * Htmp[:, 1, :, :, :]
            Htmp = np.transpose(Htmp, (0, 3, 2, 1))
            
            if np.size(H) == 0:
                H = Htmp
            else:
                H = np.concatenate((H, Htmp), axis=0)
        
        H = H.astype(np.complex64)  # 将数据转换为 complex64 类型以减小存储空间
        print(f"Finished processing Data {i+1}\n")
    
    print("All datasets have been processed.")
