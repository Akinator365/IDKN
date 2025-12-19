import os
import torch
import numpy as np
import networkx as nx
from torch_geometric.data import Data
from torch_sparse import SparseTensor
from torch_geometric.utils import add_self_loops
from Model import GDN_SIR_Predictor_Transformer_Pos


# --- 路径设置 (保持与你要求的一致) ---
BASE_DIR = os.getcwd()
REALWORLD_DATASET_PATH = os.path.join(BASE_DIR, 'data', 'networks', 'realworld')
HEATMAP_OUTPUT_PATH = os.path.join(BASE_DIR, 'data', 'heatmaps', 'realworld')

# 目标网络名称 (这里使用了你在 prompt 中提到的 _model 后缀，请确认文件名是否真的带 _model)
# 如果文件名是 karate_club_graph.txt，请去掉 _model
TARGET_NETWORK_NAME = "karate_club_graph"

# 硬编码源节点
HARDCODED_SOURCE_NODE = 1

# 构造具体路径
GRAPH_PATH = os.path.join(REALWORLD_DATASET_PATH, f"{TARGET_NETWORK_NAME}.txt")
OUTPUT_DIR = os.path.join(HEATMAP_OUTPUT_PATH, f"{TARGET_NETWORK_NAME}")


def load_graph_data(graph_path):
    """
    读取 txt 边列表并转换为 PyG Data 对象
    """
    if not os.path.exists(graph_path):
        raise FileNotFoundError(f"找不到图文件: {graph_path}")

    print(f"正在加载图数据: {graph_path}")
    # 使用 networkx 读取以处理节点 ID
    G = nx.read_edgelist(graph_path, nodetype=int)

    # 转换为 PyG 所需的 edge_index
    # 注意：PyG 需要 0-based 索引。如果你的图节点是 1-based (如 1..34)，
    # 这里我们保持原始 ID，但 PyG 内部计算通常假设节点是连续的 0..N-1。
    # 为了安全起见，我们通常需要重映射，但为了和你之前的 SIR 模拟对齐，
    # 我们假设你的 txt 文件里的节点 ID 已经是你想要的格式。

    # 构建边索引 (2, E)
    edges = list(G.edges())
    # 添加反向边 (变为无向图)
    edges += [(v, u) for u, v in edges]

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    # 获取节点数量 (取最大索引 + 1，适配 PyG)
    num_nodes = edge_index.max().item() + 1

    data = Data(edge_index=edge_index, num_nodes=num_nodes)
    return data, G.nodes()


def generate_model_heatmap(model, data, source_node_id):
    """
    利用训练好的模型权重，生成以 source_node_id 为源头的传播热力图
    (Attention Rollout)
    """
    model.eval()

    # 1. 运行一次前向传播，让模型计算出当前图结构下的所有边权重
    #    (这一步是为了填充 model.gdnX.last_alpha)
    #    [注意]：请确保 Model.py 中的 GDNConv.message 方法里已经添加了:
    #    self.last_alpha = alpha.detach()
    with torch.no_grad():
        _ = model(data)

    # 2. 准备数据
    num_nodes = data.num_nodes
    if num_nodes is None:
        num_nodes = data.x.shape[0] if data.x is not None else data.edge_index.max().item() + 1

        # 这里我们必须同样给 edge_index 添加自环，以保持维度对齐。
    edge_index_loop, _ = add_self_loops(data.edge_index, num_nodes=num_nodes)

    # 获取每一层的权重 (如果报错 AttributeError，说明 GDNConv 没存 last_alpha)
    try:
        alphas = [
            model.gdn1.last_alpha,
            model.gdn2.last_alpha,
            model.gdn3.last_alpha,
            model.gdn4.last_alpha
        ]
    except AttributeError:
        print("错误: 模型层中找不到 last_alpha 属性。")
        print("请修改 Model.py 中的 GDNConv 类，在 message 函数中添加 'self.last_alpha = alpha.detach()'")
        return None

    # 3. 开始虚拟传播 (Attention Rollout)

    # 初始化热力图状态：只有源节点是 1.0，其他是 0
    current_heatmap = torch.zeros(num_nodes, device=data.edge_index.device)

    # 边界检查
    if source_node_id >= num_nodes:
        print(f"警告: 源节点 ID {source_node_id} 超出范围 (Max {num_nodes - 1})。")
        return None

    current_heatmap[source_node_id] = 1.0

    heatmap_matrix = []  # 存储 4 步的结果

    print(f"开始生成热力图 (Source: {source_node_id})...")

    for layer_idx, alpha in enumerate(alphas):
        # 构建稀疏矩阵 (N, N)
        # SparseTensor默认是 (row, col)，这里 row=target, col=source
        adj = SparseTensor(
            row=edge_index_loop[1],  # target indices
            col=edge_index_loop[0],  # source indices
            value=alpha,
            sparse_sizes=(num_nodes, num_nodes)
        )

        # 执行传播: New_State = Adj @ Old_State
        next_heatmap = adj.matmul(current_heatmap.view(-1, 1)).squeeze()

        # [叠加模式] 模拟 SIR 的累积效应
        current_heatmap = current_heatmap + next_heatmap

        # 存下来 (转为 numpy)
        heatmap_matrix.append(current_heatmap.cpu().numpy())

    return np.array(heatmap_matrix)  # Shape: [4, Num_Nodes]


def main():
    # 1. 准备数据
    try:
        data, node_list = load_graph_data(GRAPH_PATH)
    except Exception as e:
        print(e)
        return

    # 2. 初始化模型
    print("正在初始化模型...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    hidden_dim = 64  # 这里的维度需要和你训练时保持一致
    model = GDN_SIR_Predictor_Transformer_Pos(hidden_dim=hidden_dim).to(device)
    data = data.to(device)

    # 3. [重要] 加载预训练权重
    # 如果你有训练好的 .pth 文件，请在这里加载。
    # 否则，这里使用的是随机初始化的权重，生成的热力图将是随机的，不具备物理意义。
    model_weights_path = "./training/IDKN/2025-12-19_15-16-31/checkpoint_886_epoch.pkl"
    #model_weights_path = os.path.join(BASE_DIR, 'training', 'checkpoints', f'{TARGET_NETWORK_NAME}_best.pth')

    if os.path.exists(model_weights_path):
        print(f"加载预训练权重: {model_weights_path}")
        # 加载文件
        checkpoint = torch.load(model_weights_path, map_location=device)

        # [修改] 自动判断是直接保存的 state_dict 还是 checkpoint 字典
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            # 对应你的保存方式
            print("检测到 Checkpoint 字典格式，正在提取 model_state_dict...")
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # 对应直接保存 model.state_dict() 的方式
            model.load_state_dict(checkpoint)
    else:
        print("警告: 未找到预训练权重文件，正在使用【随机初始化】权重运行！")
        print(f"请将训练好的模型保存为: {model_weights_path} 或修改脚本中的路径。")

    # 4. 生成热力图
    heatmap = generate_model_heatmap(model, data, HARDCODED_SOURCE_NODE)

    if heatmap is not None:
        # 5. 保存结果
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        # 保存 .npy
        npy_filename = os.path.join(OUTPUT_DIR, f"model_prediction_heatmap_{HARDCODED_SOURCE_NODE}.npy")
        np.save(npy_filename, heatmap)
        print(f"成功保存模型预测热力图: {npy_filename}")
        print(f"数据形状: {heatmap.shape}")

        # 同时保存节点顺序，方便后续可视化对齐
        # (因为 PyG 可能会根据 edge_index 隐式定义节点 0..N)
        # 这里简单假设 0..N 就是节点 ID，如果你的 txt ID 是离散的，需要额外的 Mapping 处理
        # node_order_path = os.path.join(OUTPUT_DIR, "model_node_order.json")
        # import json
        # with open(node_order_path, 'w') as f:
        #     json.dump(list(range(data.num_nodes)), f)


if __name__ == "__main__":
    main()