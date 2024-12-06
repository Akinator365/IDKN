import os

import networkx as nx
import numpy as np


def Generate_Graph(g_type, num_nodes, scope):
    num_min = num_nodes - scope
    num_max = num_nodes + scope
    num_nodes = np.random.randint(num_max - num_min + 1) + num_min

    if g_type == 'erdos_renyi':
        g = nx.erdos_renyi_graph(n=num_nodes, p=0.1)
    elif g_type == 'small-world':
        g = nx.connected_watts_strogatz_graph(n=num_nodes, k=4, p=0.1)
    elif g_type == 'barabasi_albert':
        g = nx.barabasi_albert_graph(n=num_nodes, m=3)
    elif g_type == 'powerlaw':
        g = nx.powerlaw_cluster_graph(n=num_nodes, m=3, p=0.05)

    g.remove_nodes_from(list(nx.isolates(g)))
    g.remove_edges_from(nx.selfloop_edges(g))
    num_nodes = len(g.nodes)

    return g, num_nodes

def GenerateTrainData(train_dataset_path, id, graph_type, num_nodes, scope):

    print(f'Generating No.{id} training {graph_type} graphs')

    data_path = os.path.join(train_dataset_path, graph_type+'_graph')
    os.makedirs(data_path, exist_ok=True)

    graph_name = f"{graph_type}_{num_nodes}_{id}.txt"

    # 查看文件是否存在，如果存在则跳过
    if os.path.exists(os.path.join(data_path, graph_name)):
        print(f"File {graph_name} already exists, skipping...")
        return

    if graph_type == 'ER':
        g_type = 'erdos_renyi'
    elif graph_type == 'WS':
        g_type = 'small-world'
    elif graph_type == 'BA':
        g_type = 'barabasi_albert'
    elif graph_type == 'PLC':
        g_type = 'powerlaw'

    # Generate Graph
    g, num_nodes = Generate_Graph(g_type, num_nodes, scope)

    # 保存图为txt文件
    # 将边写入文件
    with open(os.path.join(data_path,graph_name), 'w') as f:
        for edge in g.edges():
            f.write(f"{edge[0]} {edge[1]}\n")

    print(f"Edges saved to {data_path}\\{graph_name}")

if __name__ == '__main__':
    TRAIN_DATASET_PATH = os.path.join(os.getcwd(), 'data', 'networks', 'train')
    Synthetic_Type = ['BA', 'ER', 'PLC', 'WS']
    # 每种图的数量
    num_graph = 100
    # 图的节点数量
    num_nodes = 1000
    # 图的节点数量浮动范围
    scope = 100
    for type in Synthetic_Type:
        for id in range(num_graph):
            GenerateTrainData(TRAIN_DATASET_PATH, id, type, num_nodes, scope)
