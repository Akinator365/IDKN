import networkx as nx
import time
import json
import warnings
import numpy as np
import os


# --- 计时器（可选保留，用于统计单张图的处理耗时）---
def start_timer():
    return time.time()


def stop_timer(start_time):
    return time.time() - start_time


# --- 传播阈值计算（核心保留）---
def calculate_beta_c(G):
    """
    根据度分布的均场近似 (DBMF) 理论计算流行阈值 beta_c。
    beta_c = <k> / (<k^2> - <k>)

    :param G: networkx 图对象
    :return: (beta_c, <k>, <k^2>) 元组，如果分母为0则返回 (None, <k>, <k^2>)
    """
    num_nodes = G.number_of_nodes()
    if num_nodes == 0:
        print("图为空，无法计算。")
        return None, 0, 0

    # 1. 获取所有节点的度
    degrees = [d for n, d in G.degree()]

    # 2. 计算 <k> (平均度)
    k_avg = np.mean(degrees)

    # 3. 计算 <k^2> (度分布的二阶矩)
    k2_avg = np.mean([d ** 2 for d in degrees])

    # 4. 计算分母 (<k^2> - <k>)
    denominator = k2_avg - k_avg

    if denominator == 0:
        # 这种情况很少见，但可能发生在所有节点度都为1的图上
        warnings.warn(f"计算 beta_c 失败：分母 (<k^2> - <k>) 为 0。 (k_avg={k_avg}, k2_avg={k2_avg})")
        return None, k_avg, k2_avg

    # 5. 计算 beta_c
    beta_c = k_avg / denominator

    return beta_c, k_avg, k2_avg


# --- 读取边的函数（如果Utils中的read_edges是自定义的，这里补充一个基础实现，避免报错）---
def read_edges(graph_path):
    """
    基础的边读取函数（替代原Utils中的read_edges，若原函数有特殊逻辑可替换为自己的）
    """
    adj_list = {}
    with open(graph_path, 'r') as f:
        for line in f:
            u, v = map(int, line.strip().split())
            if u not in adj_list:
                adj_list[u] = set()
            if v not in adj_list:
                adj_list[v] = set()
            adj_list[u].add(v)
            adj_list[v].add(u)
    return adj_list


# --- 简化后的核心函数：仅计算并打印beta_c ---
def GetBetaC(graph_path, name, params):
    """
    仅计算并打印指定图的beta_c阈值，移除所有模拟逻辑
    """
    print(f"\nProcessing {name}...")
    start_time = start_timer()

    # 读取图（保留原逻辑，确保节点类型正确）
    graph = nx.read_edgelist(graph_path, nodetype=int)
    n = graph.number_of_nodes()
    if n == 0:
        print(f"  [Warning] 图 {name} 为空，跳过阈值计算。")
        return

    # 计算beta_c
    beta_c, k_avg, k2_avg = calculate_beta_c(graph)

    # 打印阈值信息（核心输出）
    print(f"  [Result] 图 {name} 的理论阈值 beta_c: {beta_c:.6f} (<k>={k_avg:.4f}, <k^2>={k2_avg:.4f})")

    # 可选：打印参数中的beta（若需要对比）
    beta = params.get('beta', 0)
    gamma = params.get('gamma', 0)
    print(f"  [Info] 参数中的 Beta: {beta:.6f} | Gamma: {gamma}")

    elapsed_time = stop_timer(start_time)
    print(f"  [Info] 处理耗时: {elapsed_time:.2f} 秒")


def GenerateBetaC(DATASET_PATH, network_params):
    """
    主协调函数，遍历所有图并计算beta_c
    """
    for network in network_params:
        params = network_params[network]
        network_type = params['type']
        print(f'\n========== 开始处理 {network} 类型图 ==========')

        entries = []
        if network_type == 'realworld':
            # Realworld 类型路径构造
            graph_path = os.path.join(DATASET_PATH, f"{network}.txt")
            entries.append((graph_path, network))
        else:
            # 合成数据集路径构造
            base_dir = f"{network_type}_graph"
            for id in range(params['num']):
                network_name = f"{network}_{id}"
                graph_path = os.path.join(DATASET_PATH, base_dir, network, f"{network_name}.txt")
                entries.append((graph_path, network_name))

        for graph_path, name in entries:
            # 检查文件是否存在
            if not os.path.exists(graph_path):
                print(f"  [Error] 图文件 {graph_path} 不存在，跳过。")
                continue
            GetBetaC(graph_path, name, params)


if __name__ == '__main__':
    # 路径配置（保留原路径逻辑）
    TRAIN_DATASET_PATH = os.path.join(os.getcwd(), 'data', 'networks', 'train')
    TEST_DATASET_PATH = os.path.join(os.getcwd(), 'data', 'networks', 'test')
    REALWORLD_DATASET_PATH = os.path.join(os.getcwd(), 'data', 'networks', 'realworld')

    # 从文件中读取参数（保留原逻辑）
    with open("Network_Parameters.json", "r") as f:
        train_network_params = json.load(f)
    with open("Network_Parameters_test.json", "r") as f:
        test_network_params = json.load(f)
    with open("Network_Parameters_realworld.json", "r") as f:
        realworld_network_params = json.load(f)

    # 执行阈值计算
    GenerateBetaC(TRAIN_DATASET_PATH, train_network_params)
    GenerateBetaC(TEST_DATASET_PATH, test_network_params)
    GenerateBetaC(REALWORLD_DATASET_PATH, realworld_network_params)