import os
import json
import networkx as nx
import numpy as np
# 新增：引入 LFR 基准图生成器
from networkx.generators.community import LFR_benchmark_graph

def Generate_Graph(g_type, network_para, scope):
    # 1. 基础节点数浮动
    num_nodes = network_para['n']
    num_min = num_nodes - scope
    num_max = num_nodes + scope
    num_nodes = np.random.randint(num_max - num_min + 1) + num_min

    # 2. 动态获取 m 值 (如果 JSON 里没写，就在 2-5 之间随机)
    # 这保证了即使是同一类图，稀疏程度也不一样
    if 'm' in network_para:
        m_val = network_para['m']
    else:
        m_val = np.random.randint(2, 6)  # 随机 m 在 [2, 5]

    g = None

    if g_type == 'erdos_renyi':
        # ER 图：p 决定密度。我们在 [0.005, 0.03] 之间浮动，避免太稀疏或太密
        # 简单估算：p = avg_k / n. 如果希望平均度在 4-10 之间
        avg_k = np.random.uniform(4, 10)
        p_val = avg_k / num_nodes
        g = nx.erdos_renyi_graph(n=num_nodes, p=p_val)

    elif g_type == 'small-world':
        # WS 小世界：重连概率 p 决定是像规则网格(p=0)还是随机图(p=1)

        # 1. 提高 k 值 (邻居数)
        # 即使是 500 节点的图，k=4 也太稀疏了。
        # 建议设置在 8 到 16 之间，且最好是偶数 (NetworkX 定义 k 为最近邻居数)
        # 这样 <k> 约为 12，理论阈值 beta_c 约为 0.08，乘 3 倍也就 0.24，非常健康
        k_val = np.random.randint(4, 9) * 2  # 生成 [8, 10, 12, 14, 16] 这样的偶数

        # 2. 拓宽 p 值 (重连概率)
        # 0.01: 极度规则，像冰糖葫芦，传播很慢，考验 GNN 对长距离的感知
        # 0.30: 接近随机图，传播极快
        p_val = np.random.uniform(0.01, 0.3)

        g = nx.connected_watts_strogatz_graph(n=num_nodes, k=k_val, p=p_val)

    elif g_type == 'barabasi_albert':
        # BA: 保持你原有的逻辑，或者稍微抖动 m
        g = nx.barabasi_albert_graph(n=num_nodes, m=m_val)

    elif g_type == 'holme_kim':
        # 【关键】HK 图：随机改变 p (聚类概率)
        # 让模型见识一下“且疏且密”的三角形结构
        p_triad = np.random.uniform(0.1, 0.9)
        g = nx.powerlaw_cluster_graph(n=num_nodes, m=m_val, p=p_triad)

    elif g_type == 'sbm':
        # 【关键】SBM：随机社区数量 + 随机连接紧密度

        # 1. 随机社区数量 (3 到 10 个社区)
        n_com = np.random.randint(3, 11)

        # 2. 分配节点
        sizes = [num_nodes // n_com] * n_com
        sizes[-1] += num_nodes % n_com

        # 3. 随机生成内部概率 p_in 和外部概率 p_out
        # 保证 p_in >> p_out 才有社区结构
        p_in = np.random.uniform(0.05, 0.2)  # 社区内部较密
        p_out = np.random.uniform(0.001, 0.01)  # 社区之间较疏

        probs = [[p_in if i == j else p_out for j in range(n_com)] for i in range(n_com)]
        g = nx.stochastic_block_model(sizes, probs)

    elif g_type == 'lfr':
        # 【关键】LFR：随机改变 mu (混合程度)
        # mu 越小，社区越好分；mu 越大，社区越乱
        mu_val = np.random.uniform(0.1, 0.4)
        avg_deg = np.random.randint(4, 10)

        try:
            g = LFR_benchmark_graph(
                n=num_nodes,
                tau1=3,
                tau2=1.5,
                mu=mu_val,  # <--- 随机化
                average_degree=avg_deg,  # <--- 随机化
                min_community=20,
                max_iters=1000,
                seed=np.random.randint(1000)
            )
            g = g.to_undirected()
        except:
            # 失败兜底
            g = nx.barabasi_albert_graph(n=num_nodes, m=m_val)

    elif g_type == 'RGG':
        # 【新增】几何图：radius 决定连通性
        # 半径越小越像铁路网，半径越大越像社交网
        # 动态计算临界半径 r_c = sqrt(ln(N)/N)，取 r_c 的 1.2~2.0 倍以保证连通
        r_c = np.sqrt(np.log(num_nodes) / num_nodes)
        radius = np.random.uniform(1.3, 2.5) * r_c
        g = nx.random_geometric_graph(n=num_nodes, radius=radius)

    # 后处理：去重边、去自环、取最大连通子图 (防止 nan)
    g.remove_edges_from(nx.selfloop_edges(g))
    if not nx.is_connected(g):
        largest_cc = max(nx.connected_components(g), key=len)
        g = g.subgraph(largest_cc).copy()

    # 重新映射节点 label 为 0..N-1 (RGG生成的是坐标，LFR可能ID不连续)
    g = nx.convert_node_labels_to_integers(g)

    return g, len(g.nodes)

def GenerateTrainData(train_dataset_path, id, network, network_para, scope):
    graph_type = network_para['type']
    graph_name = network + f'_{id}.txt'
    print(f'Generating No.{id} training {graph_type} graphs')

    data_path = os.path.join(train_dataset_path, graph_type + '_graph', network)
    os.makedirs(data_path, exist_ok=True)

    # 查看文件是否存在，如果存在则跳过
    if os.path.exists(os.path.join(data_path, graph_name)):
        print(f"File {graph_name} already exists, skipping...")
        return

    # --- 映射简写到具体的 g_type ---
    if graph_type == 'ER':
        g_type = 'erdos_renyi'
    elif graph_type == 'WS':
        g_type = 'small-world'
    elif graph_type == 'BA':
        g_type = 'barabasi_albert'
    elif graph_type == 'HK':
        g_type = 'holme_kim'
    elif graph_type == 'SBM':
        g_type = 'sbm'
    elif graph_type == 'LFR':
        g_type = 'lfr'
    else:
        # 默认回落
        g_type = 'barabasi_albert'

    # Generate Graph
    g, num_nodes = Generate_Graph(g_type, network_para, scope)

    # 保存图为txt文件
    # 将边写入文件
    with open(os.path.join(data_path, graph_name), 'w') as f:
        for edge in g.edges():
            f.write(f"{edge[0]} {edge[1]}\n")

    print(f"Edges saved to {data_path}\\{graph_name}")


if __name__ == '__main__':
    TRAIN_DATASET_PATH = os.path.join(os.getcwd(), 'data', 'networks', 'train')
    TEST_DATASET_PATH = os.path.join(os.getcwd(), 'data', 'networks', 'test')

    # 从文件中读取参数
    with open("Network_Parameters_small.json", "r") as f:
        train_network_params = json.load(f)
    # 从文件中读取参数
    with open("Network_Parameters_test.json", "r") as f:
        test_network_params = json.load(f)
    # 图的节点数量浮动范围
    scope = 100
    for network in train_network_params:
        num_graph = train_network_params[network]['num']
        for id in range(num_graph):
            GenerateTrainData(TRAIN_DATASET_PATH, id, network, train_network_params[network], scope)

    for network in test_network_params:
        num_graph = test_network_params[network]['num']
        for id in range(num_graph):
            GenerateTrainData(TEST_DATASET_PATH, id, network, test_network_params[network], scope)