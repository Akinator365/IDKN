import json
import os
import networkx as nx
import scipy
from Utils import *


def GenerateAdj(ADJ_PATH, DATASET_PATH, network_params):
    def GetAdj(graph_path, adj_path, name):
        if os.path.exists(adj_path):
            print(f"File {adj_path} already exists, skipping...")
            return
        print(f"Processing {name}")
        G = nx.read_edgelist(graph_path)
        print(G.number_of_nodes(), G.number_of_edges())
        # g_adjacent_matrix = np.array(nx.adjacency_matrix(G).todense())
        # os.makedirs(os.path.dirname(adj_path), exist_ok=True)
        # pickle_save(adj_path, g_adjacent_matrix)
        # 生成压缩稀疏行格式（CSR）的邻接矩阵
        adj_matrix = nx.adjacency_matrix(G, dtype=np.uint8)  # 使用uint8类型
        os.makedirs(os.path.dirname(adj_path), exist_ok=True)
        # 存储为压缩的稀疏矩阵（推荐）
        scipy.sparse.save_npz(adj_path, adj_matrix, compressed=True)
        print(f'pickle of {name} saved')

    for network in network_params:
        params = network_params[network]
        network_type = params['type']
        print(f'Processing {network} graphs...')

        entries = []
        if network_type == 'realworld':
            # Realworld 类型路径构造
            graph_path = os.path.join(DATASET_PATH, f"{network}.txt")
            adj_path = os.path.join(ADJ_PATH, f"{network}_adj.npz")
            entries.append((graph_path, adj_path, network))
        else:
            # 合成数据集路径构造
            base_dir = f"{network_type}_graph"
            for id in range(params['num']):
                network_name = f"{network}_{id}"
                graph_path = os.path.join(DATASET_PATH, base_dir, network, f"{network_name}.txt")
                adj_path = os.path.join(ADJ_PATH, base_dir, network, f"{network_name}_adj.npz")
                entries.append((graph_path, adj_path, network_name))

        for graph_path, adj_path, name in entries:
            GetAdj(graph_path, adj_path, name)


if __name__ == '__main__':
    TRAIN_DATASET_PATH = os.path.join(os.getcwd(), 'data', 'networks', 'train')
    TEST_DATASET_PATH = os.path.join(os.getcwd(), 'data', 'networks', 'test')
    REALWORLD_DATASET_PATH = os.path.join(os.getcwd(), 'data', 'networks', 'realworld')
    TRAIN_ADJ_PATH = os.path.join(os.getcwd(), 'data', 'adj', 'train')
    TEST_ADJ_PATH = os.path.join(os.getcwd(), 'data', 'adj', 'test')
    REALWORLD_ADJ_PATH = os.path.join(os.getcwd(), 'data', 'adj', 'realworld')

    # 从文件中读取参数
    with open("Network_Parameters.json", "r") as f:
        train_network_params = json.load(f)

    with open("Network_Parameters_test.json", "r") as f:
        test_network_params = json.load(f)

    with open("Network_Parameters_realworld.json", "r") as f:
        realworld_network_params = json.load(f)

    GenerateAdj(TRAIN_ADJ_PATH, TRAIN_DATASET_PATH, train_network_params)
    GenerateAdj(TEST_ADJ_PATH, TEST_DATASET_PATH, test_network_params)
    GenerateAdj(REALWORLD_ADJ_PATH, REALWORLD_DATASET_PATH, realworld_network_params)