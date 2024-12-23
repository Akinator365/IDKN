import json
import os

import networkx as nx
from scipy.stats import kendalltau

from Utils import read_and_sort_txt

if __name__ == '__main__':
    TRAIN_DATASET_PATH = os.path.join(os.getcwd(), 'data', 'networks', 'train')
    REALWORLD_DATASET_PATH = os.path.join(os.getcwd(), 'data', 'networks', 'realworld')
    TRAIN_LABELS_PATH = os.path.join(os.getcwd(), 'data', 'labels', 'train')
    REALWORLD_LABELS_PATH = os.path.join(os.getcwd(), 'data', 'labels', 'realworld')
    # 从文件中读取参数
    with open("Network_Parameters_small.json", "r") as f:
        network_params = json.load(f)

    test_set = ['BA_500_3', 'BA_500_5', 'BA_500_8', 'BA_1000_3', 'BA_1000_5', 'BA_1000_8']

    network_set = {}
    for network in network_params:
        network_type = network_params[network]['type']
        num_graph = network_params[network]['num']
        sum_tau_set = {}
        if network in test_set:
            print(f'Processing {network} graphs...')
            sum_tau = {'DC': 0, 'BC': 0, 'KCore': 0}
            for id in range(num_graph):
                network_name = f"{network}_{id}"
                graph_path = os.path.join(TRAIN_DATASET_PATH, network_type + '_graph', network, network_name + '.txt')
                label_path = os.path.join(TRAIN_LABELS_PATH, network_type + '_graph', network, network_name + '_labels' + ".txt")
                G = nx.read_edgelist(graph_path)
                label = read_and_sort_txt(label_path)

                # 计算度中心性
                degree_centrality = nx.degree_centrality(G)
                # 按照度中心性降序排序，得到节点列表
                sorted_dc = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)
                # 提取排序后的节点列表
                dc_list = [node for node, centrality in sorted_dc]

                # # 计算介数中心性
                # betweenness_centrality = nx.betweenness_centrality(G)
                # # 按照介数中心性降序排序
                # sorted_bc = sorted(betweenness_centrality.items(), key=lambda x: x[1], reverse=True)
                # # 提取排序后的节点列表
                # bc_list = [node for node, centrality in sorted_bc]

                # 计算节点的 k 核值
                core_numbers = nx.core_number(G)
                # 按照 k 核值降序排序
                sorted_kcore = sorted(core_numbers.items(), key=lambda x: x[1], reverse=True)
                # 提取排序后的节点列表
                kcore_list = [node for node, core in sorted_kcore]

                # 计算 Kendall's Tau 系数
                dc_tau, dc_p_value = kendalltau(label, dc_list)
                # bc_tau, bc_p_value = kendalltau(label, bc_list)
                kcore_tau, kcore_p_value = kendalltau(label, kcore_list)
                #print(f"Kendall's Tau: {tau:.4f}")
                #print(f"p-value: {p_value:.4f}")
                sum_tau['DC'] += dc_tau
                # sum_tau['BC'] += bc_tau
                sum_tau['KCore'] += kcore_tau
            sum_tau_set['DC'] = sum_tau['DC'] / num_graph
            sum_tau_set['BC'] = sum_tau['BC'] / num_graph
            sum_tau_set['KCore'] = sum_tau['KCore'] / num_graph
            network_set[network] = sum_tau_set
    print(network_set)
