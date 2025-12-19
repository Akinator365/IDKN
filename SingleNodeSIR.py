import os
import random
import networkx as nx
import time
import json
import numpy as np
from tqdm import tqdm


# --- 工具函数 ---

def start_timer():
    return time.time()


def stop_timer(start_time):
    return time.time() - start_time


def read_edges(graph_path):
    """读取边列表文件，返回邻接表 (字典形式)"""
    adj_list = {}
    with open(graph_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            u, v = int(parts[0]), int(parts[1])
            if u not in adj_list: adj_list[u] = []
            if v not in adj_list: adj_list[v] = []
            adj_list[u].append(v)
            adj_list[v].append(u)
    return adj_list


def calculate_beta_c(G):
    """计算理论阈值 (仅作参考信息)"""
    num_nodes = G.number_of_nodes()
    if num_nodes == 0: return None, 0, 0
    degrees = [d for n, d in G.degree()]
    k_avg = np.mean(degrees)
    k2_avg = np.mean([d ** 2 for d in degrees])
    denominator = k2_avg - k_avg
    if denominator == 0: return None, k_avg, k2_avg
    beta_c = k_avg / denominator
    return beta_c, k_avg, k2_avg


# --- 核心传播步进函数 ---

def sir_step(S, I, R, adj_list, beta, gamma):
    new_infected = set()
    new_recovered = set()

    # 传播：I -> S
    for i in I:
        if i in adj_list:  # 确保有邻居
            for neighbor in adj_list[i]:
                if neighbor in S and random.random() < beta:
                    new_infected.add(neighbor)

    # 恢复：I -> R
    for i in I:
        if random.random() < gamma:
            new_recovered.add(i)

    # 更新集合
    S -= new_infected
    I |= new_infected
    I -= new_recovered
    R |= new_recovered

    return S, I, R


def ic_step(S, I, R, adj_list, beta):
    new_infected = set()
    # 传播
    for i in I:
        if i in adj_list:
            for neighbor in adj_list[i]:
                if neighbor in S and random.random() < beta:
                    new_infected.add(neighbor)

    S -= new_infected
    R |= I  # IC模型中，上一轮的感染者直接视为已激活(类似于R)，不再参与传播(或取决于具体IC变种，标准IC只尝试一次)
    I = new_infected
    return S, I, R


# --- [核心修改] 生成热力图的模拟函数 ---

def simulate_heatmap(source_node, all_nodes_sorted, adj_list, beta, gamma, simulations, max_steps=4):
    """
    针对特定源节点，模拟 max_steps 步，记录每一步的全图感染状态。

    返回:
        heatmap_matrix: shape [max_steps, num_nodes]
        矩阵元素 (t, v) 表示：在时间步 t+1，节点 v 被感染(包括已恢复)的概率。
    """
    num_nodes = len(all_nodes_sorted)
    # 创建 节点ID 到 矩阵索引 的映射
    node_to_idx = {node: i for i, node in enumerate(all_nodes_sorted)}

    # 初始化计数矩阵 [时间步, 节点数]
    # counts[0] 对应 t=1, counts[1] 对应 t=2 ...
    counts = np.zeros((max_steps, num_nodes), dtype=np.float32)

    all_nodes_set = set(all_nodes_sorted)

    print(f"  [Simulating] Source: {source_node}, Steps: {max_steps}, Monte Carlo: {simulations}")

    for _ in tqdm(range(simulations), desc="Monte Carlo Loops"):
        # --- 初始化单次模拟的状态 ---
        S = set(all_nodes_set)
        I = {source_node}
        R = set()

        S.remove(source_node)

        # 用于记录本次模拟中，哪些节点已经“被波及”（Infected or Recovered）
        # 这样我们可以画出累积热力图，体现波前推进
        cumulative_infected_indices = {node_to_idx[source_node]}

        current_I = I

        # --- 按时间步推进 ---
        for t in range(max_steps):
            # 1. 运行一步传播
            if gamma == 1.0:
                S, current_I, R = ic_step(S, current_I, R, adj_list, beta)
            else:
                S, current_I, R = sir_step(S, current_I, R, adj_list, beta, gamma)

            # 2. 统计当前状态
            # 我们统计 I | R (即所有非 S 的节点) 作为 "已感染/波及"
            # 如果只统计 I，波后的中心可能会变冷，这取决于你想要 "当前活跃图" 还是 "波及范围图"
            # 通常 "热力图" 指的是传播范围，所以建议统计 I + R
            active_or_recovered = current_I | R

            # 更新本次模拟的累积感染索引
            for node in active_or_recovered:
                if node in node_to_idx:
                    cumulative_infected_indices.add(node_to_idx[node])

            # 3. 记录到总计数矩阵中
            # 对于本次模拟，所有在 cumulative_infected_indices 里的节点，在当前时间步 t 记为 1
            for idx in cumulative_infected_indices:
                counts[t, idx] += 1.0

            # 如果病毒提前灭绝 (I 为空)，后续的时间步状态与当前保持一致 (不再有新感染)
            if not current_I:
                # 这种情况下，剩下的步数 (t+1 ... max_steps-1) 状态都不变了
                # 直接把当前的累积状态加到后续所有步数里
                remaining_steps = max_steps - (t + 1)
                for next_t in range(t + 1, max_steps):
                    for idx in cumulative_infected_indices:
                        counts[next_t, idx] += 1.0
                break  # 跳出时间步循环，进行下一次 Monte Carlo

    # 计算概率：总次数 / 模拟次数
    probability_matrix = counts / simulations
    return probability_matrix


# --- 主流程 ---

def Run_Heatmap_Generation(graph_path, output_dir, params, target_source_node=1):
    print(f"---- Start Heatmap Generation for: {os.path.basename(graph_path)} ----")

    if not os.path.exists(graph_path):
        print(f"Error: Graph file not found at {graph_path}")
        return

    # 1. 读取图结构
    adj_list = read_edges(graph_path)
    graph = nx.read_edgelist(graph_path, nodetype=int)
    # 严格排序，保证矩阵的列索引与节点ID一一对应
    node_list = sorted(list(graph.nodes()))

    # 检查目标源节点是否存在
    if target_source_node not in node_list:
        print(f"[Warning] 硬编码的源节点 ID {target_source_node} 不在图中！")
        # 回退策略：使用度最大的节点作为替代，或者列表第一个
        fallback_node = node_list[0]
        print(f"[Fallback] 自动切换源节点为: {fallback_node}")
        target_source_node = fallback_node
    else:
        print(f"[Config] 选定源节点 (Source Node): {target_source_node}")

    # 2. 提取参数
    beta = params['beta']
    gamma = params['gamma']
    simulations = params['simulations']  # 建议设大一点，例如 500-1000，以获得平滑的热力图

    # 3. 运行热力图模拟 (生成 1-4 跳/步 的数据)
    MAX_STEPS = 4

    heatmap_matrix = simulate_heatmap(
        source_node=target_source_node,
        all_nodes_sorted=node_list,
        adj_list=adj_list,
        beta=beta,
        gamma=gamma,
        simulations=simulations,
        max_steps=MAX_STEPS
    )

    # 4. 保存结果
    os.makedirs(output_dir, exist_ok=True)

    # 保存 .npy 文件 (形状: [4, Num_Nodes])
    npy_filename = os.path.join(output_dir, f"heatmap_source_{target_source_node}.npy")
    np.save(npy_filename, heatmap_matrix)

    # 保存对应的节点索引顺序 (防止ID混乱)
    index_filename = os.path.join(output_dir, f"node_order.json")
    with open(index_filename, 'w') as f:
        json.dump(node_list, f)

    print(f"  [Result] Heatmap shape: {heatmap_matrix.shape}")
    print(f"  [Result] Saved heatmap to: {npy_filename}")
    print(f"  [Result] Saved node order to: {index_filename}")

    # 简单打印一下每一步的平均感染范围
    for t in range(MAX_STEPS):
        avg_infected_count = np.sum(heatmap_matrix[t])
        print(f"    Step {t + 1}: Avg Infected Count ~ {avg_infected_count:.2f} nodes")

    print("---- End Heatmap Generation ----")


if __name__ == '__main__':
    # 路径设置
    BASE_DIR = os.getcwd()
    REALWORLD_DATASET_PATH = os.path.join(BASE_DIR, 'data', 'networks', 'realworld')
    # 输出路径改为特定的 heatmap 文件夹
    HEATMAP_OUTPUT_PATH = os.path.join(BASE_DIR, 'data', 'heatmaps', 'realworld')
    PARAM_FILE = "Network_Parameters_realworld.json"

    TARGET_NETWORK_NAME = "karate_club_graph"

    # --- [关键修改] 硬编码源节点 ---
    # Karate Club 数据集中，Mr. Hi 通常是 1 或 0 (取决于数据集版本)
    # 请根据您的 karate_club_graph.txt 文件内容确认。如果是从1开始编号，这里填1。
    HARDCODED_SOURCE_NODE = 1

    # 读取参数
    if not os.path.exists(PARAM_FILE):
        print(f"Error: {PARAM_FILE} not found.")
        exit()

    with open(PARAM_FILE, "r") as f:
        all_params = json.load(f)

    if TARGET_NETWORK_NAME not in all_params:
        print(f"Error: Params for {TARGET_NETWORK_NAME} not found.")
        exit()

    target_params = all_params[TARGET_NETWORK_NAME]

    # 构造图路径
    graph_path = os.path.join(REALWORLD_DATASET_PATH, f"{TARGET_NETWORK_NAME}.txt")
    # 输出目录：专门为这个图建一个文件夹
    output_dir = os.path.join(HEATMAP_OUTPUT_PATH, f"{TARGET_NETWORK_NAME}")

    Run_Heatmap_Generation(graph_path, output_dir, target_params, target_source_node=HARDCODED_SOURCE_NODE)