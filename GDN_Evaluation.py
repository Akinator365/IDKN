import collections
import json
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import scipy as sp
import numpy as np
import torch
from matplotlib import pyplot as plt
from scipy.stats import kendalltau, rankdata
from torch_geometric.data import Data

# 引入你的 GDN 模型
from Model import GDN_SIR_Predictor, GDN_SIR_Predictor_Transformer, GDN_SIR_Predictor_Transformer_Pos, GDN_SIR_Predictor_Transformer_Pos_Residual
from Utils import sparse_adj_to_edge_index, get_logger, load_aligned_labels


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


def Evaluation(model, DATASET_PATH, ADJ_PATH, LABELS_PATH, network_params, device):
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
            label_path = os.path.join(LABELS_PATH, f"{network}_labels.txt")
            network_path = os.path.join(DATASET_PATH, f"{network}.txt")

            entries.append((network, adj_path, label_path, network_path))
        else:
            base_dir = f"{network_type}_graph"
            for id in range(params['num']):
                name = f"{network}_{id}"
                adj_path = os.path.join(ADJ_PATH, base_dir, network, f"{name}_adj.npz")
                label_path = os.path.join(LABELS_PATH, base_dir, network, f"{name}_labels.txt")
                network_path = os.path.join(DATASET_PATH, base_dir, network, f"{name}.txt")
                entries.append((name, adj_path, label_path, network_path))

        # 处理每个条目
        for name, adj_path, label_path, network_path in entries:
            if not all(os.path.exists(p) for p in [adj_path, label_path, network_path]):
                print(f"Missing files for {name}, skipping...")
                continue

            # 1. 数据加载与构建
            adj_sparse = sp.sparse.load_npz(adj_path)
            # 转换为 edge_index
            edge_index = sparse_adj_to_edge_index(adj_sparse, device=device)

            # 调用 load_aligned_labels 方法读取并对齐 TXT
            # 注意：load_aligned_labels 返回的是 CPU Tensor
            label = load_aligned_labels(network_path, label_path)

            if label is None:
                print(f"Error loading labels for {name}, skipping...")
                continue

            # 转为 Float 并移动到 device
            label = label.float().to(device)
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

            print(f"{name} | Tau: {stat:.4f} | MI: {mi:.6f} | Jaccard: {jaccard:.4f}")

    return results


def plot_results(results):
    """
    解析 results 字典并绘制两张分析图：
    1. 左图 (BA Only): X轴=参数m, 线条=不同Size, Y轴=Tau
    2. 右图 (All Types): X轴=Size, 线条=不同Type, Y轴=Tau (BA聚合均值)
    """

    # --- 1. 数据容器初始化 ---
    # 用于图1：专门存 BA 数据 -> ba_data[size][m] = [list of taus]
    ba_data = collections.defaultdict(lambda: collections.defaultdict(list))

    # 用于图2：存所有类型数据 -> overall_data[type][size] = [list of taus]
    overall_data = collections.defaultdict(lambda: collections.defaultdict(list))

    # --- 2. 统一解析逻辑 ---
    for key, val in results.items():
        # 防御性检查
        if "statistics" not in val or not val["statistics"]:
            continue

        parts = key.split('_')
        if len(parts) < 2: continue  # 格式不对跳过

        network_type = parts[0]  # BA, WS, HK...
        try:
            network_size = int(parts[1])  # 500, 1000...
        except ValueError:
            continue

        # === 动作 A: 无论什么网络，都存入 overall_data (用于图2) ===
        # 注意：这里会自动把 BA_500_3, BA_500_5 等所有 BA_500 的数据合并到一个列表中
        overall_data[network_type][network_size].extend(val["statistics"])

        # === 动作 B: 如果是 BA 网络，额外存入 ba_data (用于图1) ===
        if network_type == 'BA' and len(parts) >= 3:
            try:
                m_param = int(parts[2])  # 获取 m 参数
                ba_data[network_size][m_param].extend(val["statistics"])
            except ValueError:
                pass

    # --- 3. 开始绘图 (1行2列) ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

    # ==========================
    # === 图 1: BA 网络参数分析 ===
    # ==========================
    # 获取 BA 出现过的所有尺寸并排序
    ba_sizes = sorted(ba_data.keys())
    # 固定的 m 参数列表 (用于 X 轴顺序)
    ba_params = [3, 5, 8, 15]

    # 使用 colormap 区分不同 Size
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(ba_sizes)))

    for i, size in enumerate(ba_sizes):
        x, y = [], []
        for m in ba_params:
            vals = ba_data[size][m]
            if vals:
                x.append(m)
                y.append(np.nanmean(vals))  # 计算平均值

        if x:
            ax1.plot(x, y, marker='o', label=f"Size {size}", linewidth=2, color=colors[i])

    ax1.set_title("Analysis 1: BA Network Parameter Sensitivity", fontsize=14, fontweight='bold')
    ax1.set_xlabel("Attachment Parameter ($m$)", fontsize=12)
    ax1.set_ylabel("Kendall's Tau", fontsize=12)
    ax1.set_xticks(ba_params)
    ax1.legend(title="Network Size")
    ax1.grid(True, linestyle='--', alpha=0.6)

    # ==========================
    # === 图 2: 全网络规模对比 ===
    # ==========================
    sorted_types = sorted(overall_data.keys())
    all_sizes_seen = set()
    markers = ['o', 's', '^', 'D', 'v', 'X', 'P', '*']  # 不同形状区分类型

    for idx, net_type in enumerate(sorted_types):
        size_dict = overall_data[net_type]
        sorted_sizes = sorted(size_dict.keys())

        x, y = [], []
        for size in sorted_sizes:
            vals = size_dict[size]
            if vals:
                # 核心逻辑：这里 BA 的 y 值是所有 m 参数结果的平均值
                x.append(size)
                y.append(np.nanmean(vals))
                all_sizes_seen.add(size)

        if x:
            ax2.plot(x, y,
                     marker=markers[idx % len(markers)],
                     label=net_type,
                     linewidth=2.5,
                     alpha=0.85)

    ax2.set_title("Analysis 2: Performance vs. Scale (All Networks)", fontsize=14, fontweight='bold')
    ax2.set_xlabel("Network Size ($N$)", fontsize=12)
    ax2.set_ylabel("Kendall's Tau", fontsize=12)

    # 强制显示所有存在的 Size 刻度
    if all_sizes_seen:
        ax2.set_xticks(sorted(list(all_sizes_seen)))

    ax2.legend(title="Network Type")
    ax2.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()


def plot_realworld_results(results):
    """绘图函数 (保持原样)"""
    plt.figure(figsize=(10, 6))

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
    plt.title(f"GDN Performance on realworld Graphs")
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
    REALWORLD_RENEW_ADJ_PATH = os.path.join(os.getcwd(), 'data', 'adj', 'realworld_renew')

    TRAIN_LABELS_PATH = os.path.join(os.getcwd(), 'data', 'labels', 'train')
    TEST_LABELS_PATH = os.path.join(os.getcwd(), 'data', 'labels', 'test')
    REALWORLD_LABELS_PATH = os.path.join(os.getcwd(), 'data', 'labels', 'realworld')
    REALWORLD_RENEW_LABELS_PATH = os.path.join(os.getcwd(), 'data', 'labels', 'realworld_renew')

    TRAIN_DATASET_PATH = os.path.join(os.getcwd(), 'data', 'networks', 'train')
    TEST_DATASET_PATH = os.path.join(os.getcwd(), 'data', 'networks', 'test')
    REALWORLD_DATASET_PATH = os.path.join(os.getcwd(), 'data', 'networks', 'realworld')
    REALWORLD_RENEW_DATASET_PATH = os.path.join(os.getcwd(), 'data', 'networks', 'realworld_renew')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 2. 模型初始化
    # 必须与训练时的参数保持一致 (hidden_dim=64)
    # model = GDN_SIR_Predictor_Transformer_Pos(hidden_dim=64).to(device)
    model = GDN_SIR_Predictor_Transformer_Pos_Residual(hidden_dim=64).to(device)

    # 3. 加载 Checkpoint
    # 请替换为你训练生成的具体路径
    # 例如: "./training/GDN_Direct/2025-12-06_21-30-00/checkpoint_500_epoch.pkl"
    # good 256 all graph
    # checkpoint_path = "./training/IDKN/2025-12-25_11-27-36/checkpoint_950_epoch.pkl"
    # 学飞了
    checkpoint_path = "./training/IDKN/2025-12-31_14-37-07/checkpoint_100_epoch.pkl"

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
        # train_results = Evaluation(model, TRAIN_DATASET_PATH, TRAIN_ADJ_PATH, TRAIN_LABELS_PATH, train_params, device)
        # plot_results(train_results)

    # (B) 评估测试集 (BA Test)
    if os.path.exists("Network_Parameters_test.json"):
        print("\n--- Evaluating Test Set ---")
        with open("Network_Parameters_test.json") as f:
            test_params = json.load(f)
        # test_results = Evaluation(model, TEST_DATASET_PATH, TEST_ADJ_PATH, TEST_LABELS_PATH, test_params, device)
        # plot_results(test_results)

    # (C) 评估真实数据集 (Realworld)
    if os.path.exists("Network_Parameters_realworld.json"):
        print("\n--- Evaluating Realworld Networks ---")
        with open("Network_Parameters_realworld.json") as f:
            realworld_params = json.load(f)
        realworld_results = Evaluation(model, REALWORLD_DATASET_PATH, REALWORLD_ADJ_PATH, REALWORLD_LABELS_PATH, realworld_params, device)
        plot_realworld_results(realworld_results)

        # (D) 评估优化真实数据集 (Realworld)
        # realworld_renew_results = Evaluation(model, REALWORLD_RENEW_DATASET_PATH, REALWORLD_RENEW_ADJ_PATH, REALWORLD_RENEW_LABELS_PATH, realworld_params, device)
        # plot_realworld_results(realworld_results)