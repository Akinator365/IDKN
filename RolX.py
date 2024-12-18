import json
import os
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from graphrole import RoleExtractor, RecursiveFeatureExtractor


def save_edges_from_similarity(Network, similarity_matrix, k, file_path):
    """
    从相似度矩阵构建边并保存到文件。

    参数：
    similarity_matrix (np.array): 节点之间的相似度矩阵（n x n）。
    k (int): 每个节点连接的最相似的前 k 个节点。
    file_path (str): 保存边的文件路径。

    返回：
    None
    """
    # 确保相似度矩阵是 numpy 数组类型
    similarity_matrix = np.array(similarity_matrix)

    # 获取节点数量
    num_nodes = similarity_matrix.shape[0]

    # 用于保存所有边的列表
    edges = []

    # 遍历每个节点，选择与它最相似的 k 个节点并生成边
    for i in range(num_nodes):
        # 获取节点 i 与其他节点的相似度，忽略与自己的相似度（对角线元素）
        similarities = similarity_matrix[i]

        # 获取前 k 个最相似的节点，注意排除自己（对角线元素）
        similar_nodes = np.argsort(similarities)[::-1][1:k + 1]  # 排序并选择前 k 个（倒序）

        # 为每个相似的节点生成边
        for j in similar_nodes:
            # 保证每对边只记录一次，避免重复记录 (i, j) 和 (j, i)
            if i < j:
                edges.append((i, j))

    # 将边写入文件
    with open(os.path.join(file_path,f"{Network}_role_graph_{k}.txt"), 'w') as f:
        for edge in edges:
            f.write(f"{edge[0]} {edge[1]}\n")

    print(f"Edges saved to {file_path}\\{Network}_role_graph_{k}.txt")

def extract_features(G):
    # 获取图的节点度
    degree = dict(G.degree())
    degree_list = [degree[node] for node in G.nodes()]  # 将节点度转为列表，顺序保持一致

    # 获取图的节点聚类系数
    clustering = nx.clustering(G)
    clustering_list = [clustering[node] for node in G.nodes()]  # 将聚类系数转为列表，顺序保持一致

    # 计算每个节点在其 egonet 子图内所有节点的度数之和
    ego_degree_sum = [
        sum(degree[ego_node] for ego_node in nx.ego_graph(G, node).nodes())
        for node in G.nodes()]

    # 初始化存储其他特征的列表
    egonet_edge_ratios = []

    # 获取图中所有节点的顺序
    node_list = list(G.nodes())  # 获取图的节点列表，确保顺序一致

    # 遍历每个节点，计算特征
    for node in node_list:
        # 获取节点的 egonet 子网
        ego_subgraph = nx.ego_graph(G, node)

        # 获取子网中的边数
        egonet_edges = ego_subgraph.number_of_edges()

        # 获取子网中的节点
        ego_nodes = set(ego_subgraph.nodes())

        # 计算这些节点在原图中的实际边数
        outer_edges = 0
        ego_nodes = set(ego_subgraph.nodes())
        for n1 in ego_nodes:
            for n2 in set(G.nodes()):
                if G.has_edge(n1, n2) and n2 not in ego_nodes:
                    outer_edges += 1

        # 计算比例
        ratio = egonet_edges / (egonet_edges + outer_edges) if (egonet_edges + outer_edges) > 0 else 0  # 防止除零错误

        # 将比例加入到结果列表中
        egonet_edge_ratios.append(ratio)

    # 使用 node_list 作为顺序来确保特征的顺序与原图一致
    features = pd.DataFrame({
        'degree': degree_list,
        'clustering': clustering_list,
        'ego_degree_sum': ego_degree_sum,
        'egonet_edge_ratio': egonet_edge_ratios
    }, index=node_list)  # 设置索引为节点顺序，确保按原图顺序排列

    return features


if __name__ == '__main__':
    TRAIN_DATASET_PATH = os.path.join(os.getcwd(), 'data', 'networks', 'train')
    REALWORLD_DATASET_PATH = os.path.join(os.getcwd(), 'data', 'networks', 'realworld')
    TRAIN_ROLES_PATH = os.path.join(os.getcwd(), 'data', 'roles', 'train')
    REALWORLD_ROLES_PATH = os.path.join(os.getcwd(), 'data', 'roles', 'realworld')

    # 从文件中读取参数
    with open("Network_Parameters.json", "r") as f:
        network_params = json.load(f)

    for network in network_params:
        network_type = network_params[network]['type']
        num_graph = network_params[network]['num']
        print(f'Processing {network} graphs...')
        for id in range(num_graph):
            network_name = f"{network}_{id}"
            graph_path = os.path.join(TRAIN_DATASET_PATH, network_type + '_graph', network, network_name + '.txt')
            roles_path = os.path.join(TRAIN_ROLES_PATH, network_type + '_graph', network, network_name + "_roles.npy")
            # 如果文件已经存在，则跳过
            if os.path.exists(roles_path):
                print(f"File {roles_path} already exists, skipping...")
                continue
            else:
                print(f"Processing {network_name}")
            G = nx.read_edgelist(graph_path)
            # 提取特征
            feature_extractor = RecursiveFeatureExtractor(G)
            features = feature_extractor.extract_features()
            # 提取角色
            role_extractor = RoleExtractor(n_roles=None)
            role_extractor.extract_role_factors(features)
            
            print(f'Obtained features of {network_name}')
            #print(df)
            role_percentage = role_extractor.role_percentage
            # 将特征转换为numpy数组，并保存为npy文件
            roles_array = role_percentage.to_numpy()
            os.makedirs(os.path.dirname(roles_path), exist_ok=True)

            # 如果存在文件则跳过
            if os.path.exists(roles_path):
                print(f"File {roles_path} already exists, skipping...")
                continue
            else:
                np.save(roles_path, roles_array)


    '''
    Network = 'karate_club_graph'
    #Network = 'DNCEmails'
    network_path = os.path.join(f".\\data\\networks\\realworld\\{Network}.txt")
    role_graph_path = os.path.join(f".\\output\\")

    # 从文件加载karate_club_graph.txt
    G = nx.read_edgelist(network_path)
    # 提取特征
    feature_extractor = RecursiveFeatureExtractor(G)
    features = feature_extractor.extract_features()

    #features = extract_features(G)

    print(features)
    #对特征进行归一化
    # 对每列进行归一化，使得每列的和为 1
    #features_normalized = features.div(features.sum(axis=0), axis=1)

    # 提取角色
    role_extractor = RoleExtractor(n_roles=None)
    role_extractor.extract_role_factors(features)

    #print(role_extractor.role_percentage.round(2))
    role_percentage = role_extractor.role_percentage

    # 对role_extractor.role_percentage按照节点编号大小排序
    role_percentage.index = role_percentage.index.astype(int)  # 将索引转换为整数类型
    role_percentage = role_percentage.sort_index(axis=0)  # 按行索引升序排序
    print(role_percentage)

    # 根据role_extractor.role_percentage，计算节点之间的余弦相似度
    # 计算余弦相似度矩阵
    similarity_matrix = cosine_similarity(role_percentage)

    # 打印相似度矩阵
    print(similarity_matrix)

    save_edges_from_similarity(Network, similarity_matrix, 5, role_graph_path)

    node_roles = role_extractor.roles
'''

    # build color palette for plotting
    #unique_roles = sorted(set(node_roles.values()))
    #color_map = sns.color_palette('Paired', n_colors=len(unique_roles))
    # map roles to colors
    #role_colors = {role: color_map[i] for i, role in enumerate(unique_roles)}
    # build list of colors for all nodes in G
    #node_colors = [role_colors[node_roles[node]] for node in G.nodes]

    # plot graph
    #plt.figure()
    #with warnings.catch_warnings():
        # catch matplotlib deprecation warning
    #    warnings.simplefilter('ignore')
    #    nx.draw(
    #        G,
    #        pos=nx.spring_layout(G, seed=42),
    #        with_labels=True,
    #        node_color=node_colors,
    #    )
    #plt.show()

