import json
import os

import numpy as np
import torch
from scipy.stats import kendalltau
from torch_geometric.utils import dense_to_sparse

from AGNN_Train import load_model
from Model import CGNN, CGNN_New
from Utils import pickle_read, check_embeddings

if __name__ == '__main__':

    TRAIN_ADJ_PATH = os.path.join(os.getcwd(), 'data', 'adj', 'train')
    TEST_ADJ_PATH = os.path.join(os.getcwd(), 'data', 'adj', 'test')
    TRAIN_EMBEDDING_PATH = os.path.join(os.getcwd(), 'data', 'embedding', 'train')
    TEST_EMBEDDING_PATH = os.path.join(os.getcwd(), 'data', 'embedding', 'test')
    TRAIN_LABELS_PATH = os.path.join(os.getcwd(), 'data', 'labels', 'train')
    TEST_LABELS_PATH = os.path.join(os.getcwd(), 'data', 'labels', 'test')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 加载模型检查点
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    best = 572
    checkpoint_path = f"./training/IDKN/2025-03-05_15-47-31/checkpoint_{best}_epoch.pkl"

    model = load_model(checkpoint_path, CGNN_New, device)
    model.eval()

    # 从文件中读取参数
    with open("Network_Parameters_test.json", "r") as f:
        network_params = json.load(f)

    # 存储每类图的 Kendall tau 统计量和 p-value
    results = {}

    print("Processing graphs...")
    for network in network_params:
        network_type = network_params[network]['type']
        num_graph = network_params[network]['num']
        print(f'Processing {network} graphs...')

        # 初始化每类图的存储
        if network not in results:
            results[network] = {"statistics": [], "pvalues": []}

        for id in range(num_graph):
            network_name = f"{network}_{id}"
            adj_path = os.path.join(TEST_ADJ_PATH, network_type + '_graph', network, f'{network_name}_adj.npy')
            label_path = os.path.join(TEST_LABELS_PATH, network_type + '_graph', network,
                                      f'{network_name}_labels.npy')
            embedding_path = os.path.join(TEST_EMBEDDING_PATH, network_type + '_graph', network,
                                          f'{network_name}_embedding.npy')

            # adj_BA = pickle_read(adj_path)
            # adj_BA = torch.FloatTensor(adj_BA).to(device)

            adj_matrix = pickle_read(adj_path)

            # adj_matrix 是一个邻接矩阵，我们需要将其转为边索引格式
            edge_index = dense_to_sparse(torch.tensor(adj_matrix))[0].to(device)  # 转为 edge_index 格式

            node_feature = np.load(embedding_path)
            # check_embeddings(node_feature)
            
            # 转换为 PyTorch 张量
            node_feature = torch.FloatTensor(node_feature).to(device)
            label = np.load(label_path)
            label_t = torch.tensor(label).float().to(device)

            output = model(node_feature, edge_index)

            # 计算 Kendall tau 相关性
            stat, pval = kendalltau(output.detach().cpu().numpy(), label_t.cpu().numpy())

            print(f"{network}_{id} tau:{stat}")

            # 由于p-value很小，对其取对数
            log_pval = np.log10(pval) if pval > 0 else -100  # 避免 log(0)

            # 存储结果
            results[network]["statistics"].append(stat)
            results[network]["pvalues"].append(log_pval)

    # 计算每类图的平均 statistic 和 p-value
    for network, values in results.items():
        avg_stat = np.mean(values["statistics"])
        avg_pval = np.mean(values["pvalues"])
        print(f"\nNetwork Type: {network}")
        print(f"Average Kendall tau statistic: {avg_stat:.4f}")
        print(f"Average p-value: {avg_pval:.4f}")

