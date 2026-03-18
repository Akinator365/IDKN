import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import scipy.sparse as sp


def calculate_gcc_curve(G, ranking):
    """
    根据给定的节点排名，依次删除节点并计算 GCC 大小
    """
    G_copy = G.copy()
    n = G_copy.number_of_nodes()

    # 计算初始状态下(未删节点)的 GCC 大小
    if G_copy.number_of_nodes() > 0:
        initial_gcc = max(len(c) for c in nx.connected_components(G_copy)) / n
    else:
        initial_gcc = 0.0
    gcc_sizes = [initial_gcc]

    # 按排名依次移除节点
    for node in ranking:
        if G_copy.has_node(node):
            G_copy.remove_node(node)

        if G_copy.number_of_nodes() > 0:
            # 如果图还有节点，计算当前最大的连通块比例
            gcc = max(len(c) for c in nx.connected_components(G_copy)) / n
        else:
            gcc = 0.0

        gcc_sizes.append(gcc)

    return gcc_sizes


def main():
    print("=" * 50)
    print("Start Evaluating Network Dismantling (GCC Curve)")
    print("=" * 50)

    # 1. 配置路径
    # network_name = 'BA_1000_8_0'
    network_name = 'Figeys'

    # 原始网络邻接矩阵路径
    adj_path = os.path.join(os.getcwd(), '..', 'data', 'adj', 'realworld', f'{network_name}_adj.npz')
    # adj_path = os.path.join(os.getcwd(), '..', 'data', 'adj', 'train', 'BA_graph', 'BA_1000_8',
    #                         f'{network_name}_adj.npz')

    # ⚠️ 注意：如果以后你跑了新的时间戳，请在这里修改为你最新的模型输出文件夹
    timestamp_folder = '2026-03-12_22-29-04'
    ranking_path = os.path.join(os.getcwd(), 'training', 'RANN_Dismantling', timestamp_folder,
                                f'{network_name}_ranking.npy')

    # 2. 加载数据
    print(f"[1/4] Loading network from: {adj_path}")
    if not os.path.exists(adj_path):
        raise FileNotFoundError(f"找不到邻接矩阵文件: {adj_path}")
    adj_sparse = sp.load_npz(adj_path)
    G = nx.from_scipy_sparse_array(adj_sparse)

    print(f"[2/4] Loading RANN ranking from: {ranking_path}")
    if not os.path.exists(ranking_path):
        raise FileNotFoundError(f"找不到排名文件: {ranking_path}")
    rann_ranking = np.load(ranking_path)

    # 3. 计算 Baseline (度中心性 DC) 排名
    print("[3/4] Calculating Baseline (Degree Centrality) ranking...")
    degree_dict = dict(G.degree())
    dc_ranking = sorted(degree_dict, key=degree_dict.get, reverse=True)

    # 4. 计算下降曲线
    print("[4/4] Simulating network attacks and computing GCC curves...")
    rann_gcc = calculate_gcc_curve(G, rann_ranking)
    dc_gcc = calculate_gcc_curve(G, dc_ranking)

    # 5. 画图与保存
    print(">>> Plotting the results...")
    x_axis = np.linspace(0, 1, len(rann_gcc))

    plt.figure(figsize=(8, 6))
    plt.plot(x_axis, rann_gcc, label='Ours (RANN)', color='red', linewidth=2, marker='o')
    plt.plot(x_axis, dc_gcc, label='Degree (DC)', color='gray', linestyle='--', linewidth=2, marker='x')

    plt.title(f'Network Dismantling GCC Curve ({network_name})', fontsize=14)
    plt.xlabel('Fraction of nodes removed', fontsize=12)
    plt.ylabel('Normalized GCC size', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)

    # 自动将生成的对比图保存到 RANN 对应的文件夹下，方便写论文直接取用
    save_fig_path = os.path.join(os.getcwd(), 'training', 'RANN_Dismantling', timestamp_folder,
                                 f'{network_name}_gcc_curve.png')
    plt.savefig(save_fig_path, dpi=300, bbox_inches='tight')
    print(f"✅ Figure successfully saved to: {save_fig_path}")

    # 弹窗显示图片
    plt.show()


if __name__ == '__main__':
    main()