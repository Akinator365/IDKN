import networkx as nx
import os
import pandas as pd
import numpy as np
import time


def GetNetworkStats(DATASET_PATH, OUTPUT_PATH, RENEW_PATH, network_params):
    """
    遍历网络参数，读取图文件，计算统计指标并保存为 CSV。
    同时提取图的最大连通分量，并保存到 RENEW_PATH 中。
    """

    # 用于存储所有图的统计结果
    all_stats = []

    # 内部函数：计算单个图的指标并保存最大连通分量
    def process_single_graph(graph_path, save_path, name, category):
        if not os.path.exists(graph_path):
            print(f"[Warning] File not found: {graph_path}")
            return None

        try:
            # 1. 读取图
            # 假设图是 EdgeList 格式 (node1 node2)
            # nodetype=int 自动转换节点ID为整数
            G = nx.read_edgelist(graph_path, nodetype=int)

            # 2. 基础指标
            num_nodes = G.number_of_nodes()
            num_edges = G.number_of_edges()

            if num_nodes == 0:
                return None

            # 3. 计算连通分量
            # nx.number_connected_components 对于无向图很快
            # generator 转 list 以便多次使用
            components = list(nx.connected_components(G))
            num_components = len(components)

            # --- 新增功能：提取并保存最大连通分量 ---
            if components:
                # 找到节点数最多的那个分量
                largest_comp_nodes = max(components, key=len)
                # 创建子图副本 (必须 copy，否则修改会影响原图)
                G_lcc = G.subgraph(largest_comp_nodes).copy()

                # 确保保存目录存在
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                # 保存为 edgelist，data=False 表示不保存权重等额外信息，只保存节点对
                nx.write_edgelist(G_lcc, save_path, data=False)
                # print(f"    [Saved] LCC saved to {save_path}")
            # -------------------------------------

            # 计算每个分量的大小（节点数）
            comp_sizes = [len(c) for c in components]

            if comp_sizes:
                max_comp_size = max(comp_sizes)  # 最大连通分量大小
                min_comp_size = min(comp_sizes)  # 最小连通分量大小
            else:
                max_comp_size = 0
                min_comp_size = 0

            # 4. 计算平均度 <k> = 2E / N
            avg_degree = (2 * num_edges) / num_nodes if num_nodes > 0 else 0

            # 5. 计算网络密度 (Density) = 2E / (N*(N-1))
            density = nx.density(G)

            # 6. (可选) 平均聚类系数
            # 注意：对于超大图(百万节点+)，计算聚类系数可能比较慢。
            # 如果跑得太慢，可以把下面这行注释掉
            # avg_clustering = nx.average_clustering(G)
            avg_clustering = 0  # 暂时设为0以加快速度

            print(f"  -> {name}: N={num_nodes}, E={num_edges}, CC={num_components}")

            return {
                "Category": category,  # 网络类别 (如 BA, Realworld)
                "Name": name,  # 具体网络名
                "Nodes": num_nodes,  # 节点数
                "Edges": num_edges,  # 连边数
                "Avg_Degree": round(avg_degree, 4),  # 平均度
                "Density": round(density, 6),  # 密度
                "Components": num_components,  # 连通分量数
                "Max_Comp_Size": max_comp_size,  # 最大连通分量节点数
                "Min_Comp_Size": min_comp_size,  # 最小连通分量节点数
                "Avg_Clustering": round(avg_clustering, 4)  # 聚类系数
            }

        except Exception as e:
            print(f"[Error] Failed to process {name}: {e}")
            return None

    # --- 主循环 ---
    print(f"---- Start Analyzing Networks ----")

    for network in network_params:
        params = network_params[network]
        network_type = params['type']  # 'realworld' or 'BA', 'ER', etc.

        print(f'\nProcessing Group: {network} ({network_type})...')

        entries = []

        # --- 路径构造逻辑 (保持和你原代码一致) ---
        if network_type == 'realworld':
            # Realworld 只有一份文件
            # 路径示例: data/networks/realworld/Karate.txt
            graph_path = os.path.join(DATASET_PATH, f"{network}.txt")
            # 新增：保存路径构造
            save_path = os.path.join(RENEW_PATH, f"{network}.txt")
            entries.append((graph_path, save_path, network))
        else:
            # 合成数据集有多个 id
            # 路径示例: data/networks/train/BA_graph/BA_1000/BA_1000_0.txt
            base_dir = f"{network_type}_graph"
            for id in range(params['num']):
                # 注意：这里根据你的文件命名习惯，可能是 network_id 或 network_params_id
                # 假设是 BA_1000_0
                network_name = f"{network}_{id}"
                graph_path = os.path.join(DATASET_PATH, base_dir, network, f"{network_name}.txt")
                # 新增：保存路径构造 (保持原有目录结构)
                save_path = os.path.join(RENEW_PATH, base_dir, network, f"{network_name}.txt")
                entries.append((graph_path, save_path, network_name))

        # --- 遍历处理 ---
        for graph_path, save_path, name in entries:
            # 传入 save_path
            stats = process_single_graph(graph_path, save_path, name, network)
            if stats:
                all_stats.append(stats)

    # --- 保存结果 ---
    if all_stats:
        df = pd.DataFrame(all_stats)

        # 调整列顺序，好看一点
        cols = ["Category", "Name", "Nodes", "Edges", "Avg_Degree", "Components", "Max_Comp_Size", "Min_Comp_Size",
                "Density"]
        df = df[cols]

        output_file = os.path.join(OUTPUT_PATH, "Network_Statistics_Summary.csv")
        df.to_csv(output_file, index=False)
        print(f"\n[Success] 统计完成！结果已保存至: {output_file}")
        print(df.head())  # 打印前几行看看
    else:
        print("\n[Warning] 没有统计到任何有效数据。")


# --- 使用示例 ---
if __name__ == '__main__':
    import json

    # 路径配置 (根据你的实际路径修改)
    # 比如你要统计 Realworld 数据集
    # 原数据目录: .../data/networks/realworld
    DATASET_PATH = os.path.join(os.getcwd(), 'data', 'networks', 'realworld')

    # 新增：处理后数据的保存目录: .../data/networks/realworld_renew
    # 这里的逻辑是取 DATASET_PATH 的父目录，然后拼上 realworld_renew
    # 如果 DATASET_PATH 是 .../realworld，那么 dirname 就是 .../networks，拼接后就是 .../networks/realworld_renew
    RENEW_DATASET_PATH = os.path.join(os.getcwd(), 'data', 'networks', 'realworld_renew')

    OUTPUT_PATH = os.path.join(os.getcwd(), 'data', 'stats')  # 结果保存位置
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    print(f"Input Path: {DATASET_PATH}")
    print(f"Renew Path: {RENEW_DATASET_PATH}")

    # 读取参数配置
    # 假设你想统计 Realworld 的参数
    with open("Network_Parameters_realworld.json", "r") as f:
        realworld_params = json.load(f)

    # 运行统计 (传入 RENEW_DATASET_PATH)
    GetNetworkStats(DATASET_PATH, OUTPUT_PATH, RENEW_DATASET_PATH, realworld_params)