import json
import os
import scipy as sp
import numpy as np
import torch
from matplotlib import pyplot as plt
from scipy.stats import kendalltau, rankdata
from torch_geometric.data import Data

# 引入你的 GDN 模型
from Model import GDN_SIR_Predictor
from Utils import sparse_adj_to_edge_index, get_logger


def load_model(checkpoint_path, model, device):
    """
    加载保存的模型检查点
    """
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    # 适配保存时是 {'model_state_dict': ...} 的格式
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    return model


def jaccard_similarity(output_rank, true_rank, k=10):
    # 提取前k个元素（需处理可能的重复项）
    pred_top_k = set(np.argsort(-output_rank)[:k])
    true_top_k = set(np.argsort(-true_rank)[:k])

    intersection = len(pred_top_k & true_top_k)
    union = len(pred_top_k | true_top_k)

    return intersection / union if union != 0 else 0.0


def Evaluation(model, ADJ_PATH, LABELS_PATH, network_params, device):
    """
    GDN 专用评估函数
    区别：不需要读取 EMBEDDING_PATH，直接构建 Data 对象输入模型
    """
    results = {}

    for network in network_params:
        params = network_params[network]
        network_type = params['type']
        entries = []

        # 生成处理条目
        if network_type == 'realworld':
            adj_path = os.path.join(ADJ_PATH, f"{network}_adj.npz")
            label_path = os.path.join(LABELS_PATH, f"{network}_labels.npy")
            entries.append((network, adj_path, label_path))
        else:
            base_dir = f"{network_type}_graph"
            for id in range(params['num']):
                name = f"{network}_{id}"
                adj_path = os.path.join(ADJ_PATH, base_dir, network, f"{name}_adj.npz")
                label_path = os.path.join(LABELS_PATH, base_dir, network, f"{name}_labels.npy")
                entries.append((name, adj_path, label_path))

        # 处理每个条目
        for name, adj_path, label_path in entries:
            if not all(os.path.exists(p) for p in [adj_path, label_path]):
                print(f"Missing files for {name}, skipping...")
                continue

            # 1. 数据加载与构建
            adj_sparse = sp.sparse.load_npz(adj_path)
            # 转换为 edge_index
            edge_index = sparse_adj_to_edge_index(adj_sparse, device=device)

            label = torch.tensor(np.load(label_path)).float().to(device)
            num_nodes = label.shape[0]

            # 构建 PyG Data 对象 (GDN 需要 edge_index 和 num_nodes)
            # 不需要外部 x，模型内部会生成 initial_val
            data = Data(edge_index=edge_index, num_nodes=num_nodes)
            data = data.to(device)

            # 2. 模型推理
            with torch.no_grad():
                # GDN forward 只需要 data 对象
                output = model(data)

            # 3. 计算指标
            output_np = output.cpu().numpy().flatten()
            label_np = label.cpu().numpy().flatten()

            # --- 指标计算逻辑保持不变 ---

            # (1) Kendall's Tau
            stat, pval = kendalltau(output_np, label_np)
            log_pval = np.log10(pval) if pval > 0 else -100

            # (2) 单调性指数 (MI)
            ranks = rankdata(-output_np, method='dense')
            unique, counts = np.unique(ranks, return_counts=True)
            sum_n_alpha = np.sum(counts * (counts - 1))
            N = len(output_np)
            if N <= 1:
                mi = 1.0
            else:
                mi = (1 - sum_n_alpha / (N * (N - 1))) ** 2

            # (3) Jaccard 相似度
            percentages = [0.1, 0.2, 0.3, 0.4, 0.5]
            jaccard_scores = []
            for p in percentages:
                k = max(1, int(N * p))
                jaccard = jaccard_similarity(output_np, label_np, k)
                jaccard_scores.append(jaccard)

            # 存储结果
            if network not in results:
                results[network] = {"statistics": [], "pvalues": [], "MI": [], "Jaccard": []}
            results[network]["statistics"].append(stat)
            results[network]["pvalues"].append(log_pval)
            results[network]["MI"].append(mi)
            results[network]["Jaccard"].append(jaccard_scores)

            print(f"{name} | Tau: {stat:.4f} | MI: {mi:.4f}")

    return results


def plot_results(results, graph_type='BA'):
    """绘图函数 (保持原样)"""
    plt.figure(figsize=(10, 6))

    if graph_type == 'BA':
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
        networks = sorted(results.keys())
        data = []
        for net in networks:
            mean_stat = np.nanmean(results[net]["statistics"])
            mean_pval = np.nanmean(results[net]["pvalues"])
            data.append([
                net,
                f"{mean_stat:.4f}",
                f"{10 ** mean_pval:.2e}" if mean_pval > -100 else "N/A"
            ])

        n_rows = len(data)
        n_cols = 3
        colors = []
        header_color = '#40466e'
        colors.append([header_color] * n_cols)
        for i in range(n_rows - 1):
            color = '#F5F5F5' if i % 2 == 0 else 'white'
            colors.append([color] * n_cols)

        columns = ('Network', 'Kendall Tau', 'P-Value')
        table = plt.table(
            cellText=data,
            colLabels=columns,
            cellLoc='center',
            loc='center',
            cellColours=colors,
            colWidths=[0.3, 0.3, 0.4],
            edges='horizontal'
        )
        for (i, j), cell in table.get_celld().items():
            if i == 0:
                cell.set_text_props(color='white', weight='bold')
                cell.set_edgecolor('white')
        plt.axis('off')
        plt.title("Realworld Networks Evaluation Results", pad=20)

    plt.ylabel("Average Kendall Tau")
    plt.title(f"GDN Performance on {graph_type} Graphs")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # 1. 路径配置
    # 注意：不再需要 EMBEDDING_PATH
    TRAIN_ADJ_PATH = os.path.join(os.getcwd(), 'data', 'adj', 'train')
    TEST_ADJ_PATH = os.path.join(os.getcwd(), 'data', 'adj', 'test')
    REALWORLD_ADJ_PATH = os.path.join(os.getcwd(), 'data', 'adj', 'realworld')

    TRAIN_LABELS_PATH = os.path.join(os.getcwd(), 'data', 'labels', 'train')
    TEST_LABELS_PATH = os.path.join(os.getcwd(), 'data', 'labels', 'test')
    REALWORLD_LABELS_PATH = os.path.join(os.getcwd(), 'data', 'labels', 'realworld')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 2. 模型初始化
    # 必须与训练时的参数保持一致 (hidden_dim=64)
    model = GDN_SIR_Predictor(hidden_dim=64).to(device)

    # 3. 加载 Checkpoint
    # 请替换为你训练生成的具体路径
    # 例如: "./training/GDN_Direct/2025-12-06_21-30-00/checkpoint_500_epoch.pkl"
    checkpoint_path = "./training/IDKN/2025-12-07_20-52-37/checkpoint_1002_epoch.pkl"

    try:
        model = load_model(checkpoint_path, model, device).eval()
        print("Model loaded successfully.")
    except FileNotFoundError:
        print(f"Error: Checkpoint file not found at {checkpoint_path}")
        exit()

    # 4. 执行评估

    # (A) 评估训练集 (BA Small)
    if os.path.exists("Network_Parameters_small.json"):
        print("\n--- Evaluating Training Set (Small) ---")
        with open("Network_Parameters_small.json") as f:
            train_params = json.load(f)
        train_results = Evaluation(model, TRAIN_ADJ_PATH, TRAIN_LABELS_PATH, train_params, device)
        plot_results(train_results, graph_type='BA')

    # (B) 评估测试集 (BA Test)
    if os.path.exists("Network_Parameters_test.json"):
        print("\n--- Evaluating Test Set ---")
        with open("Network_Parameters_test.json") as f:
            test_params = json.load(f)
        test_results = Evaluation(model, TEST_ADJ_PATH, TEST_LABELS_PATH, test_params, device)
        plot_results(test_results, graph_type='BA')

    # (C) 评估真实数据集 (Realworld)
    if os.path.exists("Network_Parameters_realworld.json"):
        print("\n--- Evaluating Realworld Networks ---")
        with open("Network_Parameters_realworld.json") as f:
            realworld_params = json.load(f)
        realworld_results = Evaluation(model, REALWORLD_ADJ_PATH, REALWORLD_LABELS_PATH, realworld_params, device)
        plot_results(realworld_results, graph_type='realworld')