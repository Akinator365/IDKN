import json
import os

import numpy as np
import torch
from matplotlib import pyplot as plt
from scipy.stats import kendalltau
from torch_geometric.utils import dense_to_sparse

from AGNN_Train import load_model
from Model import CGNN, CGNN_New
from Utils import pickle_read, check_embeddings


def plot_results(results, graph_type='BA'):
    """统一绘图函数"""
    plt.figure(figsize=(10, 6))

    if graph_type == 'BA':
        # BA图参数化显示
        sizes = ["500", "1000", "2000", "5000"]
        params = [3, 5, 8, 15]

        for size in sizes:
            x, y = [], []
            for m in params:
                key = f"BA_{size}_{m}"
                if key in results and results[key]["statistics"]:
                    y.append(np.nanmean(results[key]["statistics"]))
                    x.append(m)
            if y:
                plt.plot(x, y, marker='o', label=f"Size {size}")

        plt.xlabel("BA Parameter (m)")
        plt.xticks(params)

    elif graph_type == 'realworld':
        # Realworld数据集显示
        networks = list(results.keys())
        values = [np.nanmean(results[n]["statistics"]) for n in networks]

        plt.bar(networks, values)
        plt.xlabel("Dataset")
        plt.xticks(rotation=45)

    plt.ylabel("Average Kendall Tau")
    plt.title(f"Performance on {graph_type} Graphs")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
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

    # 定义图的大小和参数
    graph_sizes = ["BA_500", "BA_1000", "BA_2000", "BA_5000"]
    graph_params = [3, 5, 8, 15]  # BA 生成参数 m

    # 初始化存储每种大小图的 tau 平均值
    tau_values = {size: {m: np.nan for m in graph_params} for size in graph_sizes}

    # 遍历 results，提取数据
    # 直接用完整的 `network_name` 去匹配 `results`
    for size in graph_sizes:
        for m in graph_params:
            network_name = f"{size}_{m}"  # 组合完整的名称
            if network_name in results:  # 直接匹配 `results` 里的 key
                if results[network_name]["statistics"]:  # 确保 statistics 不是空
                    tau_values[size][m] = np.mean(results[network_name]["statistics"])

    # 绘制折线图
    plt.figure(figsize=(8, 6))

    for size in graph_sizes:
        # 提取非 NaN 数据
        x_vals = [m for m in graph_params if not np.isnan(tau_values[size][m])]
        y_vals = [tau_values[size][m] for m in graph_params if not np.isnan(tau_values[size][m])]

        if y_vals:  # 只有在数据存在时才绘制
            plt.plot(x_vals, y_vals, marker='o', label=size)

    plt.xlabel("BA Graph Parameter (m)")
    plt.ylabel("Average Kendall Tau Statistic")
    plt.title("Kendall Tau Statistic for Different BA Graphs")
    plt.xticks(graph_params)  # 设置横坐标刻度
    plt.legend(title="Graph Size")
    plt.grid(True)
    plt.show()

