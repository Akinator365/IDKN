import os
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import sklearn

print(f"scikit-learn 版本: {sklearn.__version__}")

from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


# ============== 1. 读取嵌入向量 ==============
def load_embeddings(filename):
    """读取struct2vec输出的向量文件"""
    embeddings = {}
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    n_nodes, n_dims = map(int, lines[0].strip().split())
    print(f"节点数: {n_nodes}, 维度: {n_dims}")

    for line in lines[1:]:
        parts = list(map(float, line.strip().split()))
        node_id = int(parts[0])
        vector = np.array(parts[1:])
        embeddings[node_id] = vector

    return embeddings, n_nodes, n_dims


# ============== 2. 读取网络图 ==============
def load_network(filename):
    """读取网络文件（边列表格式）"""
    G = nx.Graph()

    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = list(map(int, line.split()))
            if len(parts) >= 2:
                u, v = parts[0], parts[1]
                G.add_edge(u, v)

    print(f"网络加载完成：{G.number_of_nodes()} 个节点，{G.number_of_edges()} 条边")
    return G


# ============== 3. 降维（优化版） ==============
def reduce_dimension(embeddings, method='tsne', n_components=2):
    """将高维向量降到2维（固定随机种子）"""
    node_ids = sorted(embeddings.keys())
    X = np.array([embeddings[node_id] for node_id in node_ids])

    print(f"正在使用 {method.upper()} 降维...")

    if method == 'tsne':
        perplexity = min(30, len(node_ids) - 1)

        # 关键优化：固定随机种子 + PCA初始化
        reducer = TSNE(
            n_components=n_components,
            random_state=42,  # 固定随机种子
            perplexity=perplexity,
            max_iter=2000,  # 增加迭代次数
            learning_rate=200,
            init='pca',  # 用PCA初始化而不是随机
            method='barnes_hut',
            early_exaggeration=12,
            metric='euclidean'
        )
    elif method == 'pca':
        reducer = PCA(n_components=n_components, random_state=42)

    X_embedded = reducer.fit_transform(X)
    return node_ids, X_embedded


# ============== 4. 可视化网络结构（优化版） ==============
def visualize_network(G, node_ids=None, X_embedded=None, save_path='network_structure.png'):
    """
    绘制原始网络结构
    使用 NetworkX 布局算法，更稳定美观
    """
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))

    # ===== 左图：Spring Layout (力导向布局) =====
    ax1 = axes[0]
    pos_spring = nx.spring_layout(G, seed=42, k=0.5, iterations=50)
    nx.draw_networkx_edges(G, pos_spring, ax=ax1, edge_color='gray', alpha=0.6, width=1.5)
    nx.draw_networkx_nodes(G, pos_spring, ax=ax1, node_color='steelblue',
                           node_size=400, alpha=0.9, edgecolors='white', linewidths=1.5)
    nx.draw_networkx_labels(G, pos_spring, ax=ax1,
                            labels={n: str(n) for n in G.nodes()},
                            font_size=9, font_color='white', font_weight='bold')
    ax1.set_title('Spring Layout (Force-Directed)', fontsize=12)
    ax1.axis('off')

    # ===== 中图：Kamada-Kawai Layout (基于距离) =====
    ax2 = axes[1]
    pos_kk = nx.kamada_kawai_layout(G)
    nx.draw_networkx_edges(G, pos_kk, ax=ax2, edge_color='gray', alpha=0.6, width=1.5)
    nx.draw_networkx_nodes(G, pos_kk, ax=ax2, node_color='coral',
                           node_size=400, alpha=0.9, edgecolors='white', linewidths=1.5)
    nx.draw_networkx_labels(G, pos_kk, ax=ax2,
                            labels={n: str(n) for n in G.nodes()},
                            font_size=9, font_color='white', font_weight='bold')
    ax2.set_title('Kamada-Kawai Layout (Distance-Based)', fontsize=12)
    ax2.axis('off')

    # ===== 右图：Circular Layout (环形布局) =====
    ax3 = axes[2]
    pos_circ = nx.circular_layout(G)
    nx.draw_networkx_edges(G, pos_circ, ax=ax3, edge_color='gray', alpha=0.6, width=1.5)
    nx.draw_networkx_nodes(G, pos_circ, ax=ax3, node_color='forestgreen',
                           node_size=400, alpha=0.9, edgecolors='white', linewidths=1.5)
    nx.draw_networkx_labels(G, pos_circ, ax=ax3,
                            labels={n: str(n) for n in G.nodes()},
                            font_size=9, font_color='white', font_weight='bold')
    ax3.set_title('Circular Layout', fontsize=12)
    ax3.axis('off')

    plt.suptitle(f'Network Structure (Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()})',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"网络结构图已保存至: {save_path}")
    plt.show()

    return pos_spring


# ============== 5. 可视化嵌入分布（优化版） ==============
def visualize_embeddings(node_ids, X_embedded, G=None, save_path='embedding_2d.png'):
    """绘制嵌入的2D分布"""
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    # 左图：纯散点图
    ax1 = axes[0]
    scatter1 = ax1.scatter(X_embedded[:, 0], X_embedded[:, 1],
                           c='steelblue', s=150, alpha=0.8,
                           edgecolors='white', linewidth=1)

    for i, node_id in enumerate(node_ids):
        ax1.annotate(str(node_id),
                     (X_embedded[i, 0], X_embedded[i, 1]),
                     fontsize=9, alpha=0.9, fontweight='bold',
                     xytext=(4, 4), textcoords='offset points',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5))

    ax1.set_title('Embedding Distribution (t-SNE 2D)', fontsize=14)
    ax1.set_xlabel('Dimension 1')
    ax1.set_ylabel('Dimension 2')
    ax1.grid(True, alpha=0.3, linestyle='--')

    # 右图：带网络结构的嵌入图
    ax2 = axes[1]
    if G is not None:
        pos = {node_ids[i]: (X_embedded[i, 0], X_embedded[i, 1]) for i in range(len(node_ids))}
        nx.draw_networkx_edges(G, pos, ax=ax2, edge_color='gray', alpha=0.4, width=1.2)
        nx.draw_networkx_nodes(G, pos, ax=ax2,
                               node_color='coral', node_size=250, alpha=0.9,
                               edgecolors='white', linewidths=1)
        nx.draw_networkx_labels(G, pos, ax=ax2,
                                labels={n: str(n) for n in G.nodes()},
                                font_size=8, font_color='black', font_weight='bold')
    ax2.set_title('Embedding with Network Structure', fontsize=14)
    ax2.set_xlabel('Dimension 1')
    ax2.set_ylabel('Dimension 2')
    ax2.grid(True, alpha=0.3, linestyle='--')

    plt.suptitle('Struct2Vec Embedding Visualization', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"嵌入图已保存至: {save_path}")
    plt.show()


# ============== 6. 聚类分析 ==============
def cluster_analysis(node_ids, X_embedded, G=None, n_clusters=4):
    """K-means聚类分析"""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
    labels = kmeans.fit_predict(X_embedded)

    print(f"\n{'=' * 60}")
    print(f"聚类结果 (k={n_clusters})")
    print(f"{'=' * 60}")

    for cluster_id in range(n_clusters):
        nodes_in_cluster = [node_ids[i] for i in range(len(node_ids)) if labels[i] == cluster_id]
        print(f"簇{cluster_id}: {len(nodes_in_cluster)}个节点 → {nodes_in_cluster}")

    # 可视化聚类结果
    plt.figure(figsize=(14, 10))
    scatter = plt.scatter(X_embedded[:, 0], X_embedded[:, 1],
                          c=labels, cmap='tab10', s=150, alpha=0.8,
                          edgecolors='white', linewidth=1.5)

    for i, node_id in enumerate(node_ids):
        plt.annotate(str(node_id),
                     (X_embedded[i, 0], X_embedded[i, 1]),
                     fontsize=9, alpha=0.9, fontweight='bold',
                     xytext=(4, 4), textcoords='offset points',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

    plt.title(f'Clustering Result (k={n_clusters})', fontsize=14)
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.colorbar(scatter, label='Cluster ID')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig('clustering_result.png', dpi=300, bbox_inches='tight')
    print(f"聚类图已保存至: clustering_result.png")
    plt.show()

    # 如果提供了网络G，分析每个簇的结构特征
    if G is not None:
        print(f"\n{'=' * 60}")
        print(f"簇结构分析")
        print(f"{'=' * 60}")
        for cluster_id in range(n_clusters):
            nodes_in_cluster = [node_ids[i] for i in range(len(node_ids)) if labels[i] == cluster_id]
            subgraph = G.subgraph(nodes_in_cluster)
            print(f"簇{cluster_id}: {len(nodes_in_cluster)}节点, "
                  f"{subgraph.number_of_edges()}条内部边, "
                  f"密度={nx.density(subgraph):.3f}")

    return labels


# ============== 7. 主程序 ==============
if __name__ == '__main__':
    # 路径配置
    VEC_PATH = os.path.abspath(os.path.join(os.getcwd(), '..', 'data', 'struct', 'emb', 'realworld'))
    NETWORK_PATH = os.path.abspath(os.path.join(os.getcwd(), '..', 'data', 'networks', 'realworld'))

    filename = 'karate_club_graph.emb'
    networkName = 'karate_club_graph.txt'

    emb_file = os.path.join(VEC_PATH, filename)
    net_file = os.path.join(NETWORK_PATH, networkName)

    print(f"{'=' * 60}")
    print(f"嵌入文件：{emb_file}")
    print(f"网络文件：{net_file}")
    print(f"{'=' * 60}\n")

    # 加载数据
    embeddings, n_nodes, n_dims = load_embeddings(emb_file)
    G = load_network(net_file)

    # 降维（固定随机种子，结果可复现）
    node_ids, X_embedded = reduce_dimension(embeddings, method='tsne')

    # 可视化网络结构（3种布局）
    visualize_network(G, save_path='network_structure.png')

    # 可视化嵌入分布
    visualize_embeddings(node_ids, X_embedded, G, save_path='embedding_2d.png')

    # 聚类分析
    cluster_labels = cluster_analysis(node_ids, X_embedded, G, n_clusters=4)

    print(f"\n{'=' * 60}")
    print(f"✅ 所有图像已生成！")
    print(f"{'=' * 60}")