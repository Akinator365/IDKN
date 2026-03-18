import os
import time
import datetime
import numpy as np
import scipy.sparse as sp
import torch
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj

# 导入你新写的 RANN 模型和 Loss
from Model import RANN, JointDismantlingLoss
from Utils import sparse_adj_to_edge_index, get_logger


def load_target_network(adj_path, features_path=None):
    """
    加载单张目标网络 (如 Route 网络)
    """
    # 1. 加载图拓扑结构
    adj_sparse = sp.load_npz(adj_path)
    edge_index = sparse_adj_to_edge_index(adj_sparse)
    num_nodes = adj_sparse.shape[0]

    # 2. 准备节点输入特征 (Struc2vec 特征)
    # 注意：这里你应该加载 4.2.1 节提取的 struc2vec 特征。
    # 如果 features_path 存在则加载，否则这里暂时用随机特征(或度特征)作为占位符示范
    if features_path and os.path.exists(features_path):
        x = torch.tensor(np.load(features_path), dtype=torch.float)
        print(f"Loaded Struc2vec features with shape: {x.shape}")
    else:
        # struc2vec 特征维度为 64
        print("Warning: No struc2vec features found. Using placeholder features.")
        x = torch.randn((num_nodes, 128), dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, num_nodes=num_nodes)
    return data, adj_sparse


if __name__ == '__main__':
    # ================= 1. 环境与日志配置 =================
    date = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    CHECKPOINTS_PATH = os.path.join(os.getcwd(), 'training', 'RANN_Dismantling', date)
    os.makedirs(CHECKPOINTS_PATH, exist_ok=True)

    logger = get_logger(os.path.join(CHECKPOINTS_PATH, 'train.log'))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # ================= 2. 加载单一目标图 (Route 网络) =================
    network_name = "Figeys"
    # network_name = "BA_1000_8_0"
    logger.info(f"Loading Target Network: {network_name} ...")

    # 替换为你实际的路径

    adj_path = os.path.join(os.getcwd(), '..', 'data', 'adj', 'realworld', f'{network_name}_adj.npz')
    features_path = os.path.join(os.getcwd(),'..', 'data', 'struct', 'vec', 'realworld', f'{network_name}_vec.npy')  # 强烈建议预先算好保存

    # adj_path = os.path.join(os.getcwd(), '..', 'data', 'adj', 'train', 'BA_graph', 'BA_1000_8', f'{network_name}_adj.npz')
    # features_path = os.path.join(os.getcwd(),'..', 'data', 'struct', 'vec', 'train', 'BA_graph', 'BA_1000_8', f'{network_name}_vec.npy')  # 强烈建议预先算好保存

    data, adj_sparse = load_target_network(adj_path, features_path)
    data = data.to(device)

    # 【关键准备】：为 Loss 函数准备稠密邻接矩阵
    # to_dense_adj 返回形状为 [1, N, N]，我们取 [0] 变成 [N, N]
    dense_adj = to_dense_adj(data.edge_index, max_num_nodes=data.num_nodes)[0].to(device)

    # ================= 3. 模型与优化器初始化 =================
    input_dim = data.x.size(1)  # struc2vec 的维度
    hidden_dim = 128

    model = RANN(input_dim=input_dim, hidden_dim=hidden_dim, heads=4, num_layers=2).to(device)

    # 初始化无监督双尺度物理 Loss
    # 超参数可以根据实际网络调整，通常让 macro 和 micro 在同一量级
    loss_fn = JointDismantlingLoss(lambda_macro=2.0, lambda_micro=0.5, alpha_l1=0.5)

    optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)

    # ================= 4. 直推式无监督训练 (Transductive Optimization) =================
    logger.info("Start Unsupervised Physics-Driven Optimization...")
    t_start = time.time()

    best_loss = float('inf')
    best_scores = None

    epochs = 1000  # 无监督单图优化通常收敛很快
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        # 1. RANN 前向传播：模型吐出所有节点的攻击分数 S_i
        scores = model(data.x, data.edge_index)

        # 2. 物理极限测试：计算宏观骨架崩塌与微观邻域粉碎损失
        total_loss, macro_loss, micro_loss, l1_loss = loss_fn(scores, dense_adj)

        # 3. 反向传播更新 QKV 权重
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # 4. 打印极其直观的物理进度
        if (epoch + 1) % 10 == 0:
            logger.info(f"Epoch [{epoch + 1:03d}/{epochs}] | "
                        f"Total: {total_loss.item():.4f} | "
                        f"Macro(λ_max): {macro_loss.item():.4f} | "
                        f"Micro(Egonet): {micro_loss.item():.4f} | "
                        f"L1(Budget): {l1_loss.item():.4f}")

        # 保存表现最好(破坏力最强)的分数配置
        if total_loss.item() < best_loss:
            best_loss = total_loss.item()
            best_scores = scores.detach().clone()

    logger.info(f"Optimization Finished in {time.time() - t_start:.2f}s!")

    # ================= 5. 输出关键节点拆解排名 =================
    logger.info("Extracting the Critical Node Ranking (Target Attack Set)...")

    # 将模型逼近出的最佳分数拉回 CPU
    final_scores = best_scores.cpu().numpy()

    # 按照分数从大到小排序，返回节点索引
    # 这些排在最前面的节点，就是大楼的“绝对承重墙”
    ranking_indices = np.argsort(final_scores)[::-1]

    logger.info(f"Top-10 Critical Nodes in {network_name}: {ranking_indices[:10]}")
    logger.info(f"Their corresponding scores: {final_scores[ranking_indices[:10]]}")

    # 保存排名结果供下游 GCC 曲线绘制使用
    np.save(os.path.join(CHECKPOINTS_PATH, f'{network_name}_ranking.npy'), ranking_indices)
    logger.info(f"Ranking saved to {CHECKPOINTS_PATH}")