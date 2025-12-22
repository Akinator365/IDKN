import os
import json
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib as mpl  # 导入 matplotlib 主包以处理 colormaps


def visualize_sir_heatmap():
    # 1. 检查文件是否存在
    if not all(os.path.exists(p) for p in [GRAPH_PATH, NPY_PATH, JSON_PATH]):
        print("Error: 缺少必要的数据文件。请先运行 SIR 模拟生成热力图。")
        return

    # 2. 加载数据
    print("Loading data...")
    G = nx.read_edgelist(GRAPH_PATH, nodetype=int)
    heatmap_matrix = np.load(NPY_PATH)
    with open(JSON_PATH, 'r') as f:
        node_order_list = json.load(f)

    # 确保图节点和数据对齐
    graph_nodes_sorted = sorted(list(G.nodes()))
    if graph_nodes_sorted != sorted(node_order_list):
        print("Warning: 图文件节点与热力图记录的节点不完全匹配，可能导致显示错误。")

    # 3. 计算并固定节点位置
    print("Calculating fixed layout...")
    pos = nx.spring_layout(G, k=0.25, iterations=50, seed=42)

    # 4. 开始绘图
    num_steps = heatmap_matrix.shape[0]
    fig, axes = plt.subplots(1, num_steps, figsize=(20, 5))
    fig.suptitle(f"SIR Cumulative Infection Probability Heatmap (Source Node: {SOURCE_NODE_ID})", fontsize=16, y=1.05)

    # [修复 1]: 使用兼容的新版 colormap 获取方式
    try:
        cmap = mpl.colormaps['Reds']
    except AttributeError:
        # 如果 matplotlib 版本极老，回退到旧方法
        cmap = plt.cm.get_cmap('Reds')

    # 设置颜色的归一化范围为 [0, 1]
    norm = mcolors.Normalize(vmin=0.0, vmax=1.0)
    # 创建一个 ScalarMappable 对象，用于手动生成 colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    print("Plotting...")
    for t in range(num_steps):
        ax = axes[t]
        current_step_probs = heatmap_matrix[t, :]

        # --- 数据对齐 ---
        prob_map = {node_id: prob for node_id, prob in zip(node_order_list, current_step_probs)}

        # 获取每个节点的概率值列表
        node_prob_values = [prob_map.get(node, 0.0) for node in G.nodes()]

        # [修复 2]: 手动将概率值映射为颜色值 (RGBA)，不再依赖 networkx 的 norm 参数
        node_colors_rgba = [cmap(norm(val)) for val in node_prob_values]

        # --- 绘制网络 ---
        nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.3, edge_color='gray')

        # 绘制节点
        # 注意：这里我们移除了 cmap 和 norm 参数，直接传入转换好的 RGBA 颜色列表
        nx.draw_networkx_nodes(G, pos, ax=ax,
                               node_color=node_colors_rgba,
                               node_size=200,
                               edgecolors='grey',
                               linewidths=0.5)

        # --- 高亮源节点 ---
        if SOURCE_NODE_ID in G:
            # 单独计算源节点的颜色
            source_color = cmap(norm(prob_map.get(SOURCE_NODE_ID, 1.0)))
            nx.draw_networkx_nodes(G, pos, ax=ax,
                                   nodelist=[SOURCE_NODE_ID],
                                   node_color=[source_color],
                                   node_size=250,
                                   edgecolors='black',
                                   linewidths=2.5)

        ax.set_title(f"Time Step {t + 1} (Hop {t + 1})")
        ax.axis('off')

        # 添加色标栏 (Colorbar)
        # 使用上面创建的 ScalarMappable 对象 sm
        plt.colorbar(sm, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)

    plt.tight_layout()

    output_img_path = os.path.join(HEATMAP_DIR, f"sir_heatmap_visualization.png")
    plt.savefig(output_img_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_img_path}")

    plt.show()


if __name__ == '__main__':
    # --- 配置路径和参数 ---
    BASE_DIR = os.getcwd()
    TARGET_NETWORK_NAME = "karate_club_graph"
    SOURCE_NODE_ID = 1  # 必须与生成热力图时使用的源节点一致

    # 输入文件路径
    GRAPH_PATH = os.path.join(BASE_DIR, 'data', 'networks', 'realworld', f"{TARGET_NETWORK_NAME}.txt")
    HEATMAP_DIR = os.path.join(BASE_DIR, 'data', 'heatmaps', 'realworld', f"{TARGET_NETWORK_NAME}")
    NPY_PATH = os.path.join(HEATMAP_DIR, f"heatmap_source_{SOURCE_NODE_ID}.npy")
    #NPY_PATH = os.path.join(HEATMAP_DIR, f"model_prediction_heatmap_{SOURCE_NODE_ID}.npy")
    JSON_PATH = os.path.join(HEATMAP_DIR, "node_order.json")

    visualize_sir_heatmap()