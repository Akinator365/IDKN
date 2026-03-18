import collections
import json
import os
import pickle
import re

import networkx as nx
from matplotlib.ticker import ScalarFormatter

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
            # 确保 results 中有 name 这个键
            if name not in results:
                results[name] = {}
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

            # 方法 B: 度中心性 Baseline
            degree_output = np.array(adj_sparse.sum(axis=1)).flatten()

            # 构建 NetworkX 图用于计算复杂指标
            # 注意：如果你的图是有向的，使用 nx.DiGraph()
            G = nx.from_scipy_sparse_array(adj_sparse)

            # 2. Betweenness Centrality (介数中心性)
            # 对于大图，k 可以设为较小值进行近似计算，这里默认全量计算
            bc_dict = nx.betweenness_centrality(G)
            bc_output = np.array([bc_dict[i] for i in range(num_nodes)])

            # 3. Eigenvector Centrality (特征向量中心性)
            try:
                ec_dict = nx.eigenvector_centrality(G, max_iter=1000)
                ec_output = np.array([ec_dict[i] for i in range(num_nodes)])
            except:
                # 如果不收敛，填 0 或使用 degree 代替
                ec_output = np.zeros(num_nodes)

            # 4. H-index (H 指数)
            # 计算逻辑：一个节点的 H 指数是指其至少有 H 个邻居的度数都大于等于 H
            ih_output = []
            for node in range(num_nodes):
                # 获取所有邻居的度数
                neighbor_degrees = [G.degree(neighbor) for neighbor in G.neighbors(node)]
                neighbor_degrees.sort(reverse=True)
                h = 0
                for i, deg in enumerate(neighbor_degrees):
                    if deg >= i + 1:
                        h = i + 1
                    else:
                        break
                ih_output.append(h)
            ih_output = np.array(ih_output)

            # 3. 统一计算并存储
            methods_to_eval = {
                "GDN_Model": output_np,
                "Degree": degree_output,
                "Betweenness": bc_output,
                "Eigenvector": ec_output,
                "H-index": ih_output
            }

            for m_name, m_pred in methods_to_eval.items():
                metrics = compute_metrics(m_pred, label_np)

                if m_name not in results[name]:
                    results[name][m_name] = {"Tau": [], "MI": [], "Jaccard": []}

                for metric_name, value in metrics.items():
                    results[name][m_name][metric_name].append(value)

            print(f"{name} | Model Tau: {results[name]['GDN_Model']['Tau'][-1]:.4f} | "
                  f"Deg Tau: {results[name]['Degree']['Tau'][-1]:.4f}")


    return results


def compute_metrics(pred_np, label_np):
    """统一计算所有指标的辅助函数"""
    N = len(pred_np)
    # (1) Kendall's Tau
    stat, pval = kendalltau(pred_np, label_np)

    # (2) 单调性指数 (MI)
    ranks = rankdata(-pred_np, method='dense')
    unique, counts = np.unique(ranks, return_counts=True)
    sum_n_alpha = np.sum(counts * (counts - 1))
    mi = (1 - sum_n_alpha / (N * (N - 1))) ** 2 if N > 1 else 1.0

    # (3) Jaccard 相似度 (取不同比例的均值或记录列表)
    percentages = [0.1, 0.2, 0.3, 0.4, 0.5]
    jaccard_scores = [jaccard_similarity(pred_np, label_np, max(1, int(N * p))) for p in percentages]

    return {"Tau": stat, "MI": mi, "Jaccard": jaccard_scores}


def plot_results(results):
    """
    1. 左图: BA 网络参数分析 (线性轴)
    2. 右图: 全网络规模对比 (对数横坐标轴)
    """
    methods = ["GDN_Model", "Degree", "H-index", "Betweenness", "Eigenvector"]
    styles = {"GDN_Model": "-", "Degree": "--", "H-index": ":", "Betweenness": "-.", "Eigenvector": "--"}
    # 定义固定颜色
    color_map = {
        "GDN_Model": "tab:red",
        "Degree": "tab:gray",
        "H-index": "tab:blue",
        "Betweenness": "tab:green",
        "Eigenvector": "tab:orange"
    }
    # ba_param_data: 用于图1 (BA m参数)
    ba_param_data = collections.defaultdict(lambda: collections.defaultdict(list))
    # comparison_data[net_type][m_name][size] = [list of taus]
    comparison_data = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(list)))

    # --- 数据解析 ---
    for name, methods_dict in results.items():
        parts = name.split('_')
        if len(parts) < 2: continue
        net_type, size = parts[0], int(parts[1])

        for m_name, metrics in methods_dict.items():
            taus = metrics.get("Tau", [])
            if not taus: continue

            # 存储所有类型、所有方法的规模对比数据
            comparison_data[net_type][m_name][size].extend(taus)

            # 专门存储 BA 的参数敏感性数据 (仅限模型)
            if net_type == 'BA' and m_name == "GDN_Model" and len(parts) >= 3:
                try:
                    m_val = int(parts[2])
                    ba_param_data[size][m_val].extend(taus)
                except:
                    pass

    # --- 2. 动态布局计算 ---
    # 图1 (BA参数) + 所有网络类型的对比图
    net_types = sorted(comparison_data.keys())
    n_plots = 2 + len(net_types)
    n_cols = 3
    n_rows = (n_plots + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8 * n_cols, 7 * n_rows))
    axes = axes.flatten()

    # 公用格式化器
    def set_log_xaxis(ax, sizes):
        ax.set_xscale('log')
        fmt = ScalarFormatter()
        fmt.set_scientific(False)
        ax.xaxis.set_major_formatter(fmt)
        if sizes:
            ax.set_xticks(sorted(list(sizes)))
        ax.grid(True, which='both', linestyle='--', alpha=0.4)

    # === 图 1: BA 网络参数分析 (保持线性) ===
    ax1 = axes[0]
    ba_sizes = sorted(ba_param_data.keys())
    ba_params = [3, 5, 8, 15]
    colors_ba = plt.cm.viridis(np.linspace(0, 0.8, len(ba_sizes)))
    for i, size in enumerate(ba_sizes):
        x, y = [], []
        for m in ba_params:
            vals = ba_param_data[size][m]
            if vals:
                x.append(m)
                y.append(np.nanmean(vals))
        if x:
            ax1.plot(x, y, marker='o', label=f"Size {size}", linewidth=2, color=colors_ba[i])
    ax1.set_title("Analysis 1: BA Parameter Sensitivity", fontsize=14, fontweight='bold')
    ax1.set_xlabel("m", fontsize=12);
    ax1.set_ylabel("Tau", fontsize=12)
    ax1.set_ylim(0.2, 1.0);
    ax1.legend();
    ax1.grid(True, linestyle='--', alpha=0.6)

    # === 图 2: 全网络规模对比 (原图2 - 模型合集) ===
    ax2 = axes[1]
    markers = ['o', 's', '^', 'D', 'v', 'X', 'P', '*']
    all_sizes = set()
    for idx, t in enumerate(net_types):
        size_dict = comparison_data[t]["GDN_Model"]
        sorted_sizes = sorted(size_dict.keys())
        x, y = [], []
        for s in sorted_sizes:
            x.append(s);
            y.append(np.nanmean(size_dict[s]));
            all_sizes.add(s)
        ax2.plot(x, y, marker=markers[idx % len(markers)], label=t, linewidth=2)
    set_log_xaxis(ax2, all_sizes)
    ax2.set_title("Analysis 2: All Networks Scale (GDN Only)", fontweight='bold')
    ax2.set_xlabel("Size (N)");
    ax2.set_ylabel("Tau");
    ax2.set_ylim(0.2, 1.0);
    ax2.legend()

    # === 图 3 及以后: 独立对比图 (Model vs Baselines) ===
    methods = ["GDN_Model", "Degree", "H-index", "Betweenness", "Eigenvector"]
    m_colors = {"GDN_Model": "tab:red", "Degree": "tab:gray", "H-index": "tab:blue",
                "Betweenness": "tab:green", "Eigenvector": "tab:orange"}

    for i, t in enumerate(net_types):
        ax = axes[i + 2]
        sizes_seen = set()
        for m_name in methods:
            if m_name not in comparison_data[t]: continue
            size_dict = comparison_data[t][m_name]
            sorted_s = sorted(size_dict.keys())
            x, y = [], []
            for s in sorted_s:
                x.append(s);
                y.append(np.nanmean(size_dict[s]));
                sizes_seen.add(s)
            ax.plot(x, y, marker='o', label=m_name, color=m_colors.get(m_name),
                    linestyle='-' if m_name == "GDN_Model" else '--')

        set_log_xaxis(ax, sizes_seen)
        ax.set_title(f"Analysis {i + 3}: {t} Comparison", fontweight='bold')
        ax.set_xlabel("Size (N)");
        ax.set_ylabel("Tau");
        ax.set_ylim(0.2, 1.0);
        ax.legend()

    # 隐藏多余子图
    for j in range(i + 3, len(axes)): axes[j].axis('off')

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
    checkpoint_path = "./training/IDKN/2025-12-31_14-37-07/checkpoint_1287_epoch.pkl"

    try:
        model = load_model(checkpoint_path, model, device).eval()
        print("Model loaded successfully.")
    except FileNotFoundError:
        print(f"Error: Checkpoint file not found at {checkpoint_path}")
        exit()

    # 使用正则表达式提取：日期时间 (2025-12-31_14-37-07) 和 Epoch (1287)
    match = re.search(r'(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}).*?checkpoint_(\d+)_epoch', checkpoint_path)
    if match:
        timestamp = match.group(1)
        epoch = match.group(2)
        test_result_filename = f"./results/result_{timestamp}_epoch{epoch}.pkl"
    else:
        # 备选方案：如果正则匹配失败，使用简单哈希或固定名
        test_result_filename = "eval_result_backup.pkl"

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
        # --- 2. 判断文件是否存在 ---
        if os.path.exists(test_result_filename):
            print(f"Found existing result file: {test_result_filename}")
            print("Loading cached results and skipping evaluation...")
            with open(test_result_filename, 'rb') as f:
                test_results = pickle.load(f)
        else:
            print(f"No cached result found. Starting Evaluation...")
            # 执行评估
            test_results = Evaluation(model, TEST_DATASET_PATH, TEST_ADJ_PATH, TEST_LABELS_PATH, test_params, device)

            # 保存结果
            with open(test_result_filename, 'wb') as f:
                pickle.dump(test_results, f)
            print(f"Evaluation finished and results saved to {test_result_filename}")
        plot_results(test_results)

    # (C) 评估真实数据集 (Realworld)
    if os.path.exists("Network_Parameters_realworld.json"):
        print("\n--- Evaluating Realworld Networks ---")
        with open("Network_Parameters_realworld.json") as f:
            realworld_params = json.load(f)
        # realworld_results = Evaluation(model, REALWORLD_DATASET_PATH, REALWORLD_ADJ_PATH, REALWORLD_LABELS_PATH, realworld_params, device)
        # plot_realworld_results(realworld_results)

        # (D) 评估优化真实数据集 (Realworld)
        # realworld_renew_results = Evaluation(model, REALWORLD_RENEW_DATASET_PATH, REALWORLD_RENEW_ADJ_PATH, REALWORLD_RENEW_LABELS_PATH, realworld_params, device)
        # plot_realworld_results(realworld_results)