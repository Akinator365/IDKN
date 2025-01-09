import os
import pickle
import logging

import numpy as np
import scipy as sp


def pickle_read(path):
    with open(path, 'rb') as file:
        data = pickle.load(file)
    return data

def pickle_save(path, data):
    with open(path, 'wb') as file:
        pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger

def min_max_normalization(x):
    min_val = x.min(axis=0, keepdims=True)  # 计算每个特征的最小值
    max_val = x.max(axis=0, keepdims=True)  # 计算每个特征的最大值
    range_val = max_val - min_val  # 计算范围

    # 如果范围为零（即所有值都相同），直接返回 0 或 其他适当的值
    range_val[range_val == 0] = 1  # 避免除以零
    return (x - min_val) / range_val

# 定义一个函数来读取文件并按第二列降序排列
def read_and_sort_txt(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.split()
            index = int(parts[0])  # 第一列
            value = float(parts[1])  # 第二列
            data.append((index, value))

    # 按第二列的值进行降序排列
    sorted_data = sorted(data, key=lambda x: x[1], reverse=True)
    sorted_indexes = [x[0] for x in sorted_data]

    return sorted_indexes

def read_edges(file_path):
    adj_list = {}

    # 打开并读取文件
    with open(file_path, 'r') as f:
        for line in f:
            # 每行是一个边，格式为 "node1 node2"
            node1, node2 = map(int, line.strip().split())

            # 如果节点1不在字典中，初始化它
            if node1 not in adj_list:
                adj_list[node1] = set()
            # 如果节点2不在字典中，初始化它
            if node2 not in adj_list:
                adj_list[node2] = set()

            # 将每条边加到两个节点的邻接表中
            adj_list[node1].add(node2)
            adj_list[node2].add(node1)

    return adj_list


def analyze_npy_files(folder_path):
    """
    读取指定文件夹下的所有 .npy 文件，并输出每个文件的数据行数和列数。

    :param folder_path: 文件夹路径
    """
    # 获取文件夹中所有 .npy 文件
    files = [f for f in os.listdir(folder_path) if f.endswith('.npy')]

    if not files:
        print("指定文件夹中没有 .npy 文件")
        return

    # 遍历每个文件
    for file_name in files:
        file_path = os.path.join(folder_path, file_name)

        # 加载 .npy 文件
        data = np.load(file_path)

        # 获取行数和列数
        if len(data.shape) == 2:  # 确保是二维数组
            rows, cols = data.shape
        elif len(data.shape) == 1:  # 一维数组
            rows, cols = data.shape[0], 1
        else:
            print(f"文件 {file_name} 不是二维或一维数据，跳过...")
            continue

        # 输出结果
        print(f"文件: {file_name} | 行数: {rows} | 列数: {cols}")

def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)
