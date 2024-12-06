import os

import networkx as nx
import numpy as np
import pandas as pd

from Utils import pickle_save, pickle_read


def extract_graph_features(graph: nx.Graph) -> pd.DataFrame:
    """
    提取图的度中心性、介数中心性、紧密度中心性、特征向量中心性、k-shell、PageRank并返回DataFrame。

    Parameters:
    graph (nx.Graph): 输入的networkx图

    Returns:
    pd.DataFrame: 各个节点的中心性特征组成的DataFrame
    """
    # 计算度中心性
    degree_centrality = nx.degree_centrality(graph)

    # 计算介数中心性
    betweenness_centrality = nx.betweenness_centrality(graph)

    # 计算紧密度中心性
    closeness_centrality = nx.closeness_centrality(graph)

    # 计算特征向量中心性
    #eigenvector_centrality = nx.eigenvector_centrality(graph)

    # 计算k-shell
    k_shell = nx.core_number(graph)  # 返回每个节点的k-shell值

    # 计算PageRank
    pagerank = nx.pagerank(graph)  # 返回每个节点的PageRank值

    # 计算CI
    collective_influence = calculate_ci_with_l_neighbors(graph, 2)

    # 计算节点的聚类系数
    node_clustering = nx.clustering(graph)

    # 创建一个字典，包含所有中心性特征
    centrality_data = {
        'degree_centrality': degree_centrality,
        'betweenness_centrality': betweenness_centrality,
        'closeness_centrality': closeness_centrality,
        #'eigenvector_centrality': eigenvector_centrality,
        'k_shell': k_shell,
        'pagerank': pagerank,
        'collective_influence': collective_influence,
        'node_clustering': node_clustering,
    }

    # 将字典转换为DataFrame
    centrality_df = pd.DataFrame(centrality_data)

    return centrality_df


def calculate_ci_with_l_neighbors(graph: nx.Graph, l: int) -> dict:
    """
    计算图中每个节点的集体影响力 (CI)，考虑l阶邻居。

    CI(i) = (k_i - 1) * Σ (k_l - 1), 其中k_l是l阶邻居的度数

    参数：
    graph (nx.Graph): 输入的图
    l (int): 邻居阶数

    返回：
    dict: 各个节点的集体影响力 (CI)
    """
    ci_dict = {}

    for node in graph.nodes():
        # 获取节点的度数 k_i
        k_i = graph.degree(node)

        # 获取节点的l阶邻居
        # 使用networkx提供的single_source_shortest_path_length方法计算l阶邻居
        l_neighbors = set()
        for neighbor in nx.single_source_shortest_path_length(graph, node, cutoff=l):
            if neighbor != node:  # 排除节点自身
                l_neighbors.add(neighbor)

        # 计算集体影响力
        ci_sum = 0
        for neighbor in l_neighbors:
            k_neighbor = graph.degree(neighbor)
            ci_sum += (k_neighbor - 1)

        # 计算节点的集体影响力
        ci_dict[node] = (k_i - 1) * ci_sum

    return ci_dict


def corehd(graph: nx.Graph):
    """
    计算图的 CoreHD 值，并返回一个由多个树组成的森林。

    参数：
    graph (nx.Graph): 输入的图

    返回：
    list: 一个包含多个树的森林，每棵树表示一个子图。
    """
    # Step 1: 获取2-core
    core = nx.core_number(graph)
    two_core_nodes = {node for node, core_num in core.items() if core_num >= 2}

    # Step 2: 初始化森林（一个空的子图列表）
    forest = []

    while two_core_nodes:
        # 找到2-core中度数最大的节点
        max_degree_node = max(two_core_nodes, key=lambda node: graph.degree(node))

        # 移除节点，并更新2-core
        graph.remove_node(max_degree_node)
        core = nx.core_number(graph)

        # 更新当前2-core中的节点集合
        two_core_nodes = {node for node, core_num in core.items() if core_num >= 2}

        # 如果2-core为空，停止循环
        if not two_core_nodes:
            break

        # 记录当前的连通子图作为森林的一部分
        # 对图进行连通性分割，找到所有树
        connected_components = list(nx.connected_components(graph))
        forest.extend([graph.subgraph(component).copy() for component in connected_components])

    return forest


if __name__ == '__main__':
    TRAIN_DATASET_PATH = os.path.join(os.getcwd(), 'data', 'networks', 'train')
    REALWORLD_DATASET_PATH = os.path.join(os.getcwd(), 'data', 'networks', 'realworld')
    TRAIN_FEATURES_PATH = os.path.join(os.getcwd(), 'data', 'features', 'train')
    REALWORLD_FEATURES_PATH = os.path.join(os.getcwd(), 'data', 'features', 'realworld')
    TRAIN_ADJ_PATH = os.path.join(os.getcwd(), 'data', 'adj', 'train')
    REALWORLD_ADJ_PATH = os.path.join(os.getcwd(), 'data', 'adj', 'realworld')

    Synthetic_Type = ['BA', 'ER', 'PLC', 'WS']
    num_graph = 10
    for type in Synthetic_Type:
        print(f'Processing {type} graphs...')
        for id in range(num_graph):
            network_name = f"{type}_1000_{id}"
            graph_path = os.path.join(TRAIN_DATASET_PATH, type + '_graph', network_name + '.txt')
            G = nx.read_edgelist(graph_path)
            # 提取特征并打印
            df = extract_graph_features(G)
            print(f'Obtained features of {network_name}')
            #print(df)

            # 将特征转换为numpy数组，并保存为npy文件
            feature_array = df.to_numpy()
            feature_path = os.path.join(TRAIN_FEATURES_PATH, type + '_graph', network_name + "_features.npy")
            os.makedirs(os.path.dirname(feature_path), exist_ok=True)
            np.save(feature_path, feature_array)

            g_adjacent_matrix = np.array(nx.adjacency_matrix(G).todense())
            adj_path = os.path.join(TRAIN_ADJ_PATH, type + '_graph', network_name + "_adj.npy")
            os.makedirs(os.path.dirname(adj_path), exist_ok=True)
            pickle_save(adj_path, g_adjacent_matrix)
            print(f'pickle of {network_name} saved')




