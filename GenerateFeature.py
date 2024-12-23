import json
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

    # 计算PageRank
    pagerank = nx.pagerank(graph)  # 返回每个节点的PageRank值

    # 计算CI
    collective_influence = calculate_ci_with_l_neighbors(graph, 4)

    # 计算节点的聚类系数
    node_clustering = nx.clustering(graph)

    # 计算节点的H-index
    h_index = compute_h_index(graph)

    # 计算节点的CoreHD
    core_hd = corehd(graph)

    # 计算节点的二阶邻居数
    high_order_nodes = Counts_High_Order_Nodes(graph, 2)

    # 计算节点的一阶邻居的平均度
    average_degree = nx.average_neighbor_degree(graph)


    # 创建一个字典，包含所有中心性特征
    centrality_data = {
        'degree_centrality': degree_centrality,
        'betweenness_centrality': betweenness_centrality,
        'closeness_centrality': closeness_centrality,
        'pagerank': pagerank,
        'collective_influence': collective_influence,
        'node_clustering': node_clustering,
        'h_index': h_index,
        'core_hd': core_hd,
        'high_order_nodes': high_order_nodes,
        'average_degree': average_degree
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
    计算每个节点的 CoreHD 值。

    参数：
    graph (nx.Graph): 输入的图

    返回：
    dict: 包含每个节点 CoreHD 值的字典 {节点: CoreHD 值}
    """
    # Step 1: 计算初始的 2-core
    core = nx.core_number(graph)
    two_core_nodes = {node for node, core_num in core.items() if core_num >= 2}

    # Step 2: 初始化结果字典和临时图
    corehd_values = {node: 0 for node in graph.nodes()}  # 存储每个节点的 CoreHD 值
    temp_graph = graph.copy()  # 复制图用于操作

    # Step 3: 逐步移除节点并计算 CoreHD 值
    while two_core_nodes:
        # 在当前 2-core 中找到度数最大的节点
        max_degree_node = max(two_core_nodes, key=lambda node: temp_graph.degree(node))
        max_degree = temp_graph.degree(max_degree_node)

        # 记录该节点的 CoreHD 值
        corehd_values[max_degree_node] = max_degree

        # 移除该节点
        temp_graph.remove_node(max_degree_node)

        # 重新计算 2-core 节点集合
        core = nx.core_number(temp_graph)
        two_core_nodes = {node for node, core_num in core.items() if core_num >= 2}

    return corehd_values

def Counts_High_Order_Nodes(G, depth = 2):
    NODES_LIST = list(G.nodes)
    output = {}
    output = output.fromkeys(NODES_LIST)
    for node in NODES_LIST:
        layers = dict(nx.bfs_successors(G, source=node, depth_limit=depth))
        high_order_nodes = sum([len(i) for i in layers.values()])
        output[node] = high_order_nodes

    return output


def compute_h_index(graph):
    """
    计算网络中每个节点的 H-index。

    参数:
    - graph: NetworkX 图

    返回:
    - h_index_dict: 包含每个节点 H-index 的字典
    """
    h_index_dict = {}  # 存储每个节点的 H-index

    for node in graph.nodes():
        # 获取节点的所有邻居的度数
        neighbors = list(graph.neighbors(node))
        neighbor_degrees = [graph.degree(neighbor) for neighbor in neighbors]

        # 排序邻居度数，从大到小
        neighbor_degrees.sort(reverse=True)

        # 计算 H-index
        h = 0
        for i, degree in enumerate(neighbor_degrees):
            if degree >= i + 1:  # 满足 H-index 条件
                h = i + 1
            else:
                break

        h_index_dict[node] = h  # 保存节点的 H-index

    return h_index_dict


if __name__ == '__main__':
    TRAIN_DATASET_PATH = os.path.join(os.getcwd(), 'data', 'networks', 'train')
    REALWORLD_DATASET_PATH = os.path.join(os.getcwd(), 'data', 'networks', 'realworld')
    TRAIN_FEATURES_PATH = os.path.join(os.getcwd(), 'data', 'features', 'train')
    REALWORLD_FEATURES_PATH = os.path.join(os.getcwd(), 'data', 'features', 'realworld')

    # 从文件中读取参数
    with open("Network_Parameters.json", "r") as f:
        network_params = json.load(f)

    for network in network_params:
        network_type = network_params[network]['type']
        num_graph = network_params[network]['num']
        print(f'Processing {network} graphs...')
        for id in range(num_graph):
            network_name = f"{network}_{id}"
            feature_path = os.path.join(TRAIN_FEATURES_PATH, network_type + '_graph', network, network_name + "_features.npy")

            # 如果文件已经存在，则跳过
            if os.path.exists(feature_path):
                print(f"File {feature_path} already exists, skipping...")
                continue
            else:
                print(f"Processing {network_name}")

            graph_path = os.path.join(TRAIN_DATASET_PATH, network_type + '_graph', network, network_name + '.txt')
            G = nx.read_edgelist(graph_path)
            # 提取特征并打印
            df = extract_graph_features(G)
            print(f'Obtained features of {network_name}')
            #print(df)
            os.makedirs(os.path.dirname(feature_path), exist_ok=True)
            # 将特征转换为numpy数组，并保存为npy文件
            features_array = df.to_numpy()
            np.save(feature_path, features_array)




