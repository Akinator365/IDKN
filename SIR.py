"""
程序主要功能
输入：网络图邻接矩阵，需要被设置为感染源的节点序列，感染率，免疫率，迭代次数step
输出：被设置为感染源的节点序列的SIR感染情况---每次的迭代结果（I+R）/n
"""
import random

import networkx as nx
import numpy as np
import scipy as sp
from scipy.sparse import coo_matrix
import torch


def update_node_status(graph, node, beta, gamma):
    """
    更新节点状态
    :param graph: 网络图
    :param node: 节点序数
    :param beta: 感染率
    :param gamma: 免疫率
    """
    # 如果当前节点状态为 感染者(I) 有概率gamma变为 免疫者(R)

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
                    graph.nodes[node]['status1'] = 'I'
                    break
    if graph.nodes[node]['status'] == 'I':
        p = random.random()
        if p < gamma:
            graph.nodes[node]['status1'] = 'R'
def update(graph):
    for node in graph:
        graph.nodes[node]['status'] = graph.nodes[node]['status1']

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


def SIR_network(graph, source, beta, gamma):
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
        graph.nodes[node]['status'] = 'S'
        graph.nodes[node]['status1'] = 'S  '# 将所有节点的状态设置为 易感者（S）
    # 设置初始感染源

    graph.nodes[source]['status'] = 'I'
    graph.nodes[source]['status1'] = 'I  '# 将感染源序列中的节点设置为感染源，状态设置为 感染者（I）
    # 记录初始状态
    sir_values.append(1 / n)
    # 开始迭代感染
    s0 = n -1
    i0 =1
    r0 =0
    while 1:
        # 针对对每个节点进行状态更新以完成本次迭代
        for node in graph:
            update_node_status(graph, node, beta, gamma)  # 针对node号节点进行SIR过程
        update(graph)
        s, i, r = count_node(graph)  # 得到本次迭代结束后各个状态（S、I、R）的节点数目
        if s== s0 and i == i0 and r == r0:
            break
        else:
            s0 = s
            i0 = i
            r0 = r
        sir = (i + r) / n  # 该节点的sir值为迭代结束后 感染节点数i+免疫节点数r
        sir_values.append(sir)  # 将本次迭代的sir值加入数组
    return i0 + r0

def caculate(adj):
    degree = np.zeros(adj.shape[0])
    for i in range(adj.shape[0]):
        degree[i] = adj[i].sum()

    sum_1 = 0
    sum_2 = 0
    for i in range(adj.shape[0]):
        sum_1 += degree[i]
        sum_2 += degree[i] * degree[i]
    avg1 = sum_1 / adj.shape[0]
    avg2 = sum_2 / adj.shape[0]

    return avg1 / (avg2 - avg1)

def caculate_avg(graph,beta,node):
    sum=0
    for i in range(100):
        sum+=SIR_network(graph,node,beta,1)
    return sum/100

if __name__ == '__main__':
    edge = np.genfromtxt("./moreno_oz/out.moreno_oz_oz", delimiter=" ",
                         dtype=int)
    edge = edge - 1
    print(edge)
    edges = edge[:, :2]
    print(edges)
    print(edges.max())
    adj = coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(217, 217))
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj_oz = torch.FloatTensor(np.array(adj.todense()))
    adj = np.array(adj_oz)
    beta = caculate(adj) * 1.5
    print(beta)
    beta = 0.0001
    graph = nx.from_numpy_array(adj)
    print(graph.number_of_nodes())
    print(graph.number_of_edges())

    lable_oz = [caculate_avg(graph, beta, i) for i in range(adj.shape[0])]
    print(lable_oz)
    np.save('lable_oz.npy', lable_oz)