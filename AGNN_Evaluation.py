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


def Evaluation(model, ADJ_PATH, LABELS_PATH, EMBEDDING_PATH, network_params, device):
    """统一评估函数，支持所有数据集类型"""
    results = {}

    for network in network_params:
        params = network_params[network]
        network_type = params['type']
        entries = []

        # 生成处理条目
        if network_type == 'realworld':
            adj_path = os.path.join(ADJ_PATH, f"{network}_adj.npy")
            label_path = os.path.join(LABELS_PATH, f"{network}_labels.npy")
            embedding_path = os.path.join(EMBEDDING_PATH, f"{network}_embedding.npy")
            entries.append((network, adj_path, label_path, embedding_path))
        else:
            base_dir = f"{network_type}_graph"
            for id in range(params['num']):
                name = f"{network}_{id}"
                adj_path = os.path.join(ADJ_PATH, base_dir, network, f"{name}_adj.npy")
                label_path = os.path.join(LABELS_PATH, base_dir, network, f"{name}_labels.npy")
                embedding_path = os.path.join(EMBEDDING_PATH, base_dir, network, f"{name}_embedding.npy")
                entries.append((name, adj_path, label_path, embedding_path))

        # 处理每个条目
        for name, adj_path, label_path, embedding_path in entries:
            if not all(os.path.exists(p) for p in [adj_path, label_path, embedding_path]):
                print(f"Missing files for {name}, skipping...")
                continue

            # 数据加载
            adj_matrix = pickle_read(adj_path)
            edge_index = dense_to_sparse(torch.tensor(adj_matrix))[0].to(device)
            node_feature = torch.FloatTensor(np.load(embedding_path)).to(device)
            label = torch.tensor(np.load(label_path)).float().to(device)

            # 模型推理
            with torch.no_grad():
                output = model(node_feature, edge_index)

            # 计算指标
            stat, pval = kendalltau(output.cpu().numpy(), label.cpu().numpy())
            log_pval = np.log10(pval) if pval > 0 else -100

            # 存储结果
            if network not in results:
                results[network] = {"statistics": [], "pvalues": []}
            results[network]["statistics"].append(stat)
            results[network]["pvalues"].append(log_pval)
            print(f"{name} tau:{stat:.4f}")

    return results

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
        # 表格展示真实网络结果
        networks = sorted(results.keys())
        data = []

        # 准备表格数据
        for net in networks:
            mean_stat = np.nanmean(results[net]["statistics"])
            mean_pval = np.nanmean(results[net]["pvalues"])
            data.append([
                net,
                f"{mean_stat:.4f}",
                f"{10 ** mean_pval:.2e}" if mean_pval > -100 else "N/A"
            ])

        # 创建颜色数组（修复点）
        n_rows = len(data)
        n_cols = 3
        colors = []
        header_color = '#40466e'  # 深蓝色表头
        colors.append([header_color] * n_cols)  # 表头颜色

        # 数据行颜色（斑马条纹）
        for i in range(n_rows - 1):
            color = '#F5F5F5' if i % 2 == 0 else 'white'
            colors.append([color] * n_cols)

        # 创建表格
        columns = ('Network', 'Kendall Tau', 'P-Value')
        table = plt.table(
            cellText=data,
            colLabels=columns,
            cellLoc='center',
            loc='center',
            cellColours=colors,  # 使用修正后的颜色数组
            colWidths=[0.3, 0.3, 0.4],
            edges='horizontal'  # 只显示水平分割线
        )

        # 设置表头样式
        for (i, j), cell in table.get_celld().items():
            if i == 0:  # 表头行
                cell.set_text_props(color='white', weight='bold')
                cell.set_edgecolor('white')

        # 隐藏坐标轴
        plt.axis('off')
        plt.title("Realworld Networks Evaluation Results", pad=20)

    plt.ylabel("Average Kendall Tau")
    plt.title(f"Performance on {graph_type} Graphs")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':

    # 初始化路径
    TRAIN_ADJ_PATH = os.path.join(os.getcwd(), 'data', 'adj', 'train')
    TEST_ADJ_PATH = os.path.join(os.getcwd(), 'data', 'adj', 'test')
    REALWORLD_ADJ_PATH = os.path.join(os.getcwd(), 'data', 'adj', 'realworld')
    TRAIN_EMBEDDING_PATH = os.path.join(os.getcwd(), 'data', 'embedding', 'train')
    TEST_EMBEDDING_PATH = os.path.join(os.getcwd(), 'data', 'embedding', 'test')
    REALWORLD_EMBEDDING_PATH = os.path.join(os.getcwd(), 'data', 'embedding', 'realworld')
    TRAIN_LABELS_PATH = os.path.join(os.getcwd(), 'data', 'labels', 'train')
    TEST_LABELS_PATH = os.path.join(os.getcwd(), 'data', 'labels', 'test')
    REALWORLD_LABELS_PATH = os.path.join(os.getcwd(), 'data', 'labels', 'realworld')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 加载模型检查点
    best = 572
    checkpoint_path = f"./training/IDKN/2025-03-05_15-47-31/checkpoint_{best}_epoch.pkl"

    # 加载模型和参数
    model = load_model(checkpoint_path, CGNN_New, device).eval()

    # 评估测试集
    with open("Network_Parameters_test.json") as f:
        test_params = json.load(f)
    test_results = Evaluation(model, TEST_ADJ_PATH, TEST_LABELS_PATH, TEST_EMBEDDING_PATH, test_params, device)
    plot_results(test_results, graph_type='BA')

    # 评估realworld数据集
    with open("Network_Parameters_realworld.json") as f:
        realworld_params = json.load(f)
    realworld_results = Evaluation(model, REALWORLD_ADJ_PATH, REALWORLD_LABELS_PATH,
                                   REALWORLD_EMBEDDING_PATH, realworld_params, device)
    plot_results(realworld_results, graph_type='realworld')

