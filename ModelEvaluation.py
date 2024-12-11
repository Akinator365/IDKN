import os

import networkx as nx
import numpy as np
import torch
from scipy.stats import kendalltau

from Model import *
from Utils import *
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse

def load_model(checkpoint_path, model_class, device):
    """
    加载保存的模型检查点
    :param checkpoint_path: 模型文件的路径 (.pkl 文件)
    :param model_class: 定义的模型类，例如 IDKN
    :param device: 设备（'cuda' 或 'cpu'）
    :return: 加载权重的模型实例
    """
    # 初始化模型
    model = model_class().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)  # 加载检查点
    model.load_state_dict(checkpoint['model_state_dict'])  # 加载模型参数
    return model


if __name__ == '__main__':
    TRAIN_DATASET_PATH = os.path.join(os.getcwd(), 'data', 'networks', 'train')
    REALWORLD_DATASET_PATH = os.path.join(os.getcwd(), 'data', 'networks', 'realworld')
    TRAIN_FEATURES_PATH = os.path.join(os.getcwd(), 'data', 'features', 'train')
    REALWORLD_FEATURES_PATH = os.path.join(os.getcwd(), 'data', 'features', 'realworld')
    TRAIN_ADJ_PATH = os.path.join(os.getcwd(), 'data', 'adj', 'train')
    REALWORLD_ADJ_PATH = os.path.join(os.getcwd(), 'data', 'adj', 'realworld')
    TRAIN_LABELS_PATH = os.path.join(os.getcwd(), 'data', 'labels', 'train')

    # network_name = 'DNCEmails'
    # network_name = 'karate_club_graph'
    network_type = 'BA'
    network_name = 'BA_1000_0'
    graph_path = os.path.join(TRAIN_DATASET_PATH, network_type + '_graph', network_name + '.txt')
    feature_path = os.path.join(TRAIN_FEATURES_PATH, network_type + '_graph', network_name + "_features.npy")
    adj_path = os.path.join(TRAIN_ADJ_PATH, network_type + '_graph', network_name + "_adj.npy")
    label_path = os.path.join(TRAIN_LABELS_PATH, network_type + '_graph', network_name + '_labels' + ".txt")
    # 加载模型检查点
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    best = 92
    checkpoint_path = f"./training/IDKN/checkpoint_{best}_epoch.pkl"
    model = load_model(checkpoint_path, IDKN, device)
    model.eval()  # 设置模型为评估模式


    # 加载真实网络的邻接矩阵、节点特征
    adj_matrix = pickle_read(adj_path)
    node_features = np.load(feature_path)

    # 转换为 edge_index 格式
    edge_index = dense_to_sparse(torch.tensor(adj_matrix))[0]  # 转为 edge_index 格式

    # 对节点特征进行归一化（如果需要）
    node_features = min_max_normalization(node_features)

    # 创建 Data 对象
    data = Data(x=torch.tensor(node_features, dtype=torch.float),
                edge_index=edge_index,
                num_nodes=node_features.shape[0])

    # 将数据移到同样的设备
    data = data.to(device)

    # 使用模型预测
    with torch.no_grad():  # 关闭梯度计算
        predictions = model(data.x, data.x, data.edge_index, data.num_nodes)

    # 输出预测结果
    #print("Predictions:", predictions)

    G = nx.read_edgelist(graph_path)

    # 获得图的节点列表
    node_list = list(G.nodes)

    # 对齐编号和预测值
    output_dict = {node: val.item() for node, val in zip(node_list, predictions)}

    # 打印对齐结果
    for node, val in output_dict.items():
        print(f"{node}\t{val:.4f}")

    label = read_and_sort_txt(label_path)
    # 按预测值降序排列节点
    predict = sorted(output_dict.items(), key=lambda x: x[1], reverse=True)
    predict_sorted_indexes = [int(node) for node, _ in predict]

    # 计算 Kendall's Tau 系数
    tau, p_value = kendalltau(label, predict_sorted_indexes)
    print(f"Kendall's Tau: {tau:.4f}")
    print(f"p-value: {p_value:.4f}")

    #print("Label\tPredict")
    #for lbl, pred in zip(label, predict_sorted_indexes):
    #    print(f"{lbl}\t{pred}")

