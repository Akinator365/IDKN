import os
import random
import networkx as nx
import numpy as np
import time
from multiprocessing import Pool  # 导入并行计算池

# 计时器
def start_timer():
    return time.time()

def stop_timer(start_time):
    return time.time() - start_time

# 并行化的 SIR 计算过程
def parallel_simulation(node, graph, simulations, beta, gamma, step):
    count = 0
    for sim in range(simulations):
        inf = sum(SIR_network(graph, [node], beta, gamma, step))
        count += inf
    aveinf = count / simulations
    return node, aveinf  # 返回节点ID和对应的平均影响力

#SIR
def update_node_status(graph, node, beta, gamma):
    """
    更新节点状态
    :param graph: 网络图
    :param node: 节点序数
    :param beta: 感染率
    :param gamma: 免疫率
    """
    # 如果当前节点状态为 感染者(I) 有概率gamma变为 免疫者(R)
    if graph.nodes[node]['status'] == 'I':
        p = random.random()
        if p < gamma:
            graph.nodes[node]['status'] = 'R'
    # 如果当前节点状态为 易感染者(S) 有概率beta变为 感染者(I)
    if graph.nodes[node]['status'] == 'S':
        # 获取当前节点的邻居节点
        # 无向图：G.neighbors(node)
        # 有向图：G.predecessors(node)，前驱邻居节点，即指向该节点的节点；G.successors(node)，后继邻居节点，即该节点指向的节点。
        neighbors = list(graph.neighbors(node))
        # 对当前节点的邻居节点进行遍历
        for neighbor in neighbors:
            # 邻居节点中存在 感染者(I)，则该节点有概率被感染为 感染者(I)
            if graph.nodes[neighbor]['status'] == 'I':
                p = random.random()
                if p < beta:
                    graph.nodes[node]['status'] = 'I'
                    break


def count_node(graph):
    """
    计算当前图内各个状态节点的数目
    :param graph: 输入图
    :return: 各个状态（S、I、R）的节点数目
    """
    s_num, i_num, r_num = 0, 0, 0
    for node in graph:
        if graph.nodes[node]['status'] == 'S':
            s_num += 1
        elif graph.nodes[node]['status'] == 'I':
            i_num += 1
        else:
            r_num += 1
    return s_num, i_num, r_num

def SIR_network(graph, source, beta, gamma, step):
    """
    获得感染源的节点序列的SIR感染情况
    :param graph: networkx创建的网络
    :param source: 需要被设置为感染源的节点Id所构成的序列
    :param beta: 感染率
    :param gamma: 免疫率
    :param step: 迭代次数
    """
    n = graph.number_of_nodes()  # 网络节点个数
    sir_values = []  # 存储每一次迭代后网络中感染节点数I+免疫节点数R的总和
    # 初始化节点状态
    for node in graph:
        graph.nodes[node]['status'] = 'S'  # 将所有节点的状态设置为 易感者（S）
    # 设置初始感染源
    for node in source:
        graph.nodes[node]['status'] = 'I'  # 将感染源序列中的节点设置为感染源，状态设置为 感染者（I）
    # 记录初始状态
    sir_values.append(len(source) / n)
    # 开始迭代感染
    for s in range(step):
        # 针对对每个节点进行状态更新以完成本次迭代
        for node in graph:
            update_node_status(graph, node, beta, gamma)  # 针对node号节点进行SIR过程
        s, i, r = count_node(graph)  # 得到本次迭代结束后各个状态（S、I、R）的节点数目
        sir = (i + r) / n  # 该节点的sir值为迭代结束后 感染节点数i+免疫节点数r
        sir_values.append(sir)  # 将本次迭代的sir值加入数组
    return sir_values


def SIR_Single(graph, graph_name):
    print("---- start creating labels ----")
    simulations = 1000
    degree = dict(nx.degree(graph))
    # 平均度为所有节点度之和除以总节点数
    ave_degree =  sum(degree.values()) / len(graph)
    # 计算节点的二阶平均度
    second_order_avg_degree = []
    for node in graph.nodes():
        neighbors = list(graph.neighbors(node))
        second_order_degrees = [graph.degree(neighbor) for neighbor in neighbors]
        if len(second_order_degrees) > 0:
            second_order_avg_degree.append(sum(second_order_degrees) / len(second_order_degrees))  # 计算邻居节点的平均度
    # 计算二阶平均度
    second_order_avg_degree = sum(second_order_avg_degree) / len(second_order_avg_degree)
    beta = ave_degree / second_order_avg_degree # 感染率
    gamma = 0.1  # 免疫率
    step = 20  # 迭代次数
    influence = list()
    for i in graph.nodes:
        print(f"Node {i}")
        count = 0
        for sim in range(simulations):
            inf = sum(SIR_network(graph,[i], beta, gamma, step))
            count += inf
        aveinf = count / simulations
        influence.append(aveinf)
    # print(graph.nodes)
    print(influence)

    output_dir = os.path.join(".", "data", "labels")
    # 输出图中节点的顺序和对应的影响力
    for idx, node in enumerate(graph.nodes):
        print(f"Node {node}: Influence {influence[idx]}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 创建并打开文件，写入影响力数据
    txt_filename = os.path.join(output_dir, graph_name + ".txt")

    with open(txt_filename, "w") as f:
        for idx, node in enumerate(graph.nodes):
            # 格式化输出，node_id 和 influence
            f.write(f"{node}\t{influence[idx]}\n")

    print(f"Influence values saved to {txt_filename}")
    print("---- end creating labels ----")


def SIR_Multiple(graph, label_path):
    print("---- start creating labels ----")
    simulations = 1000
    degree = dict(nx.degree(graph))
    ave_degree = sum(degree.values()) / len(graph)
    second_order_avg_degree = []
    for node in graph.nodes():
        neighbors = list(graph.neighbors(node))
        second_order_degrees = [graph.degree(neighbor) for neighbor in neighbors]
        if len(second_order_degrees) > 0:
            second_order_avg_degree.append(sum(second_order_degrees) / len(second_order_degrees))
    second_order_avg_degree = sum(second_order_avg_degree) / len(second_order_avg_degree)
    beta = ave_degree / second_order_avg_degree
    gamma = 0.1
    step = 20

    # 使用 multiprocessing Pool 来并行计算每个节点的影响力
    # Pool(processes=num_threads) 可指定并发线程数量
    with Pool() as pool:
        results = pool.starmap(parallel_simulation, [(i, graph, simulations, beta, gamma, step) for i in graph.nodes])

    # 将影响力存储在字典中
    influence = {}
    for node, aveinf in results:
        influence[node] = aveinf

    #for node in graph.nodes:
    #    print(f"Node {node}: Influence {influence[node]}")

    # 创建并打开文件，写入影响力数据
    txt_filename = label_path + ".txt"

    with open(txt_filename, "w") as f:
        for node in graph.nodes:
            f.write(f"{node}\t{influence[node]}\n")

    print(f"Influence values saved to {txt_filename}")
    print("---- end creating labels ----")


def Conver_to_Array(labels_path):
    # 读取label，转换为array
    with open(labels_path + '.txt', "r") as f:
        lines = f.readlines()
        labels = np.array([float(line.strip().split("\t")[1]) for line in lines])
    #print(labels)
    np.save(labels_path + '.npy', labels)


if __name__ == '__main__':
    TRAIN_DATASET_PATH = os.path.join(os.getcwd(), 'data', 'networks', 'train')
    REALWORLD_DATASET_PATH = os.path.join(os.getcwd(), 'data', 'networks', 'realworld')
    TRAIN_LABELS_PATH = os.path.join(os.getcwd(), 'data', 'labels', 'train')
    REALWORLD_LABELS_PATH = os.path.join(os.getcwd(), 'data', 'labels', 'realworld')
    Synthetic_Type = ['BA', 'ER', 'PLC', 'WS']
    # 每种图的数量
    num_graph = 100
    # 图的节点数量
    num_nodes = 1000
    # 图的节点数量浮动范围
    for type in Synthetic_Type:
        print(f'Processing {type} graphs...')
        for id in range(num_graph):
            network_name = f"{type}_{num_nodes}_{id}"
            graph_path = os.path.join(TRAIN_DATASET_PATH, type + '_graph', network_name + '.txt')
            labels_path = os.path.join(TRAIN_LABELS_PATH, type + '_graph', network_name + "_labels")
            os.makedirs(os.path.dirname(labels_path), exist_ok=True)
            txt_filepath = labels_path + ".txt"
            # 如果文件已经存在，则跳过
            if os.path.exists(txt_filepath):
                print(f"File {txt_filepath} already exists, skipping...")
                continue
            else:
                print(f"Processing {network_name}")

            G = nx.read_edgelist(graph_path)
            start_time = start_timer()  # 记录开始时间
            # SIR_Single(G, Network)
            SIR_Multiple(G, labels_path)
            elapsed_time = stop_timer(start_time)  # 计算函数运行时间
            print(f"Total time taken: {elapsed_time:.2f} seconds")
            Conver_to_Array(labels_path)