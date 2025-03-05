import json
import os
import networkx as nx
import numpy as np
from Utils import *


def GenerateAdj(ADJ_PATH, DATASET_PATH, network_params):
    for network in network_params:
        network_type = network_params[network]['type']
        num_graph = network_params[network]['num']
        print(f'Processing {network} graphs...')
        for id in range(num_graph):
            network_name = f"{network}_{id}"
            adj_path = os.path.join(ADJ_PATH, network_type + '_graph', network, network_name + "_adj.npy")

            # 如果文件已经存在，则跳过
            if os.path.exists(adj_path):
                print(f"File {adj_path} already exists, skipping...")
                continue
            else:
                print(f"Processing {network_name}")

            graph_path = os.path.join(DATASET_PATH, network_type + '_graph', network, network_name + '.txt')
            G = nx.read_edgelist(graph_path)
            g_adjacent_matrix = np.array(nx.adjacency_matrix(G).todense())
            os.makedirs(os.path.dirname(adj_path), exist_ok=True)
            pickle_save(adj_path, g_adjacent_matrix)
            print(f'pickle of {network_name} saved')


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

    GenerateAdj(TRAIN_ADJ_PATH, TRAIN_DATASET_PATH, train_network_params)
    GenerateAdj(TEST_ADJ_PATH, TEST_DATASET_PATH, test_network_params)