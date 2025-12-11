import datetime
import json
import os
import random
import sys
import time
import scipy as sp
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import torch.optim as optim
from Model import CGNN_New, CGNN_GAT, GDN_SIR_Predictor, GDN_SIR_Predictor_Transformer, GDN_SIR_Predictor_JK_Attention, \
    GDN_SIR_Predictor_Transformer_Pos
from Utils import pickle_read, get_logger, sparse_adj_to_edge_index
from torch_geometric.utils import to_dense_batch

DEFAULT_EPS = 1e-10
PADDED_Y_VALUE = -1


def listMLE_batch(y_pred, y_true, eps=DEFAULT_EPS, padded_value_indicator=PADDED_Y_VALUE):
    """
    向量化版本的 ListMLE，支持 Batch 并行计算。
    输入形状: [batch_size, max_nodes]
    """
    # 随机打乱以打破平局 (Random tie breaking)
    # y_pred: [B, N], y_true: [B, N]
    batch_size, n_nodes = y_pred.shape
    random_indices = torch.randperm(n_nodes)

    y_pred_shuffled = y_pred[:, random_indices]
    y_true_shuffled = y_true[:, random_indices]

    # 对真实标签进行排序
    y_true_sorted, indices = y_true_shuffled.sort(descending=True, dim=-1)

    # 识别填充值 (Padding)
    mask = y_true_sorted == padded_value_indicator

    # 根据真实标签的顺序重排预测值
    preds_sorted_by_true = torch.gather(y_pred_shuffled, dim=1, index=indices)

    # 将填充位置的预测值设为负无穷，使其在 softmax 中不起作用
    preds_sorted_by_true[mask] = float("-inf")

    # 数值稳定性处理：减去最大值
    max_pred_values, _ = preds_sorted_by_true.max(dim=1, keepdim=True)
    preds_sorted_by_true_minus_max = preds_sorted_by_true - max_pred_values

    # Clip 防止溢出
    preds_sorted_by_true_minus_max = torch.clamp(preds_sorted_by_true_minus_max, min=-100, max=100)

    # 计算 Cumsum (ListMLE 核心)
    # flip 是为了从后往前加 (Likelihood of being ranked 1st, given remaining items)
    cumsums = torch.cumsum(preds_sorted_by_true_minus_max.exp().flip(dims=[1]), dim=1).flip(dims=[1])

    # Log Likelihood
    observation_loss = torch.log(cumsums + eps) - preds_sorted_by_true_minus_max

    # 掩盖填充值的 Loss
    observation_loss[mask] = 0.0

    # 返回每张图的总 Loss (形状: [Batch_Size])
    # 注意：这里我们不求 mean，只求 sum，方便外部做归一化
    return torch.sum(observation_loss, dim=1)


def save_model(model, optimizer, epoch, path):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch
    }
    path_checkpoint = os.path.join(path, "checkpoint_{}_epoch.pkl".format(epoch))
    torch.save(checkpoint, path_checkpoint)

if __name__ == '__main__':

    TRAIN_ADJ_PATH = os.path.join(os.getcwd(), 'data', 'adj', 'train')
    TRAIN_LABELS_PATH = os.path.join(os.getcwd(), 'data', 'labels', 'train')
    adj_path = TRAIN_ADJ_PATH
    labels_path = TRAIN_LABELS_PATH

    date = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    CHECKPOINTS_PATH = os.path.join(os.getcwd(), 'training', 'IDKN', date)
    os.makedirs(CHECKPOINTS_PATH, exist_ok=True)

    # 初始化日志记录器
    LOGGING_PATH = os.path.join(CHECKPOINTS_PATH, 'train.log')
    # 传入 logger
    IDKN_logger = get_logger(LOGGING_PATH)
    sys.stdout = IDKN_logger  # 让 print() 也写入日志

    # 从文件中读取参数
    with open("Network_Parameters_small.json", "r") as f:
        network_params = json.load(f)

    data_list = []  # 用于存储多个图的数据
    IDKN_logger.info("Processing graphs...")
    for network in network_params:
        network_type = network_params[network]['type']
        num_graph = network_params[network]['num']
        IDKN_logger.info(f'Processing {network} graphs...')
        for id in range(num_graph):
            network_name = f"{network}_{id}"
            single_adj_path = os.path.join(adj_path, network_type + '_graph', network, network_name + '_adj.npz')
            single_labels_path = os.path.join(labels_path, network_type + '_graph', network, network_name + '_labels.npy')

            adj_sparse = sp.sparse.load_npz(single_adj_path)  # 加载压缩稀疏矩阵
            edge_index = sparse_adj_to_edge_index(adj_sparse) # 转换为边索引

            labels = np.load(single_labels_path)
            y = torch.tensor(labels, dtype=torch.float)

            # 创建 PyTorch Geometric 的 Data 对象
            data = Data(edge_index=edge_index, y=y, num_nodes=len(labels))

            # 将图数据添加到 data_list 中
            data_list.append(data)

    train_loader = DataLoader(data_list, batch_size=8, shuffle=True)

    random.seed(17)
    np.random.seed(17)
    torch.manual_seed(17)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    IDKN_logger.info(f"using device:{device}")

    # 初始化模型
    # num_nodes_max 不是必须的，因为输入特征是广播的，但可以用来设定 hidden_dim
    model = GDN_SIR_Predictor_Transformer_Pos(hidden_dim=64).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

    # 使用 ReduceLROnPlateau (智能调度)
    # mode='min': 监测指标越小越好 (Loss)
    # factor=0.5: 每次调整为原来的 0.5 倍
    # patience=50: 如果 50 个 epoch 内 Loss 没有明显下降(threshold)，就触发调整
    # min_lr=1e-6: 学习率下限
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.9,
        patience=80,
        min_lr=1e-5
    )

    t_total = time.time()
    best_loss = float('inf')
    bad_counter = 0
    best_epoch = 0

    IDKN_logger.info("Start Training GDN Direct Predictor...")

    for epoch in range(3000):
        model.train()
        total_loss = 0

        for data in train_loader:
            data = data.to(device)

            # 1. 前向传播 (GDN 依旧处理稀疏图，这是最高效的)
            # out: [Total_Nodes]
            out = model(data)

            # 2. 【关键优化】将稀疏输出转换为 Dense Batch
            # batch_idx: [Total_Nodes] -> 指示每个节点属于哪个图
            # out_dense: [Batch_Size, Max_Nodes]
            # mask: [Batch_Size, Max_Nodes] -> 指示哪些位置是真实节点，哪些是填充
            out_dense, mask = to_dense_batch(out, data.batch)
            y_dense, _ = to_dense_batch(data.y, data.batch)

            # 3. 处理 Padding
            # 将填充位置的标签设为 -1 (PADDED_Y_VALUE)，这样 listMLE 就会忽略它们
            # ~mask 表示填充的位置
            y_dense[~mask] = PADDED_Y_VALUE

            # 4. 一次性计算整个 Batch 的 Loss
            # graph_losses: [Batch_Size]
            graph_losses = listMLE_batch(out_dense, y_dense)

            # 5. 归一化 (除以每张图的实际节点数)
            # mask.sum(dim=1) 得到每张图的真实节点数
            num_nodes_per_graph = mask.sum(dim=1).float()
            # 避免除以 0 (虽然理论上不会有空图)
            num_nodes_per_graph = torch.clamp(num_nodes_per_graph, min=1.0)

            normalized_losses = graph_losses / num_nodes_per_graph

            # 6. 取 Batch 平均
            loss = normalized_losses.mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

        # Calculate average loss for the epoch
        avg_loss = total_loss / len(train_loader)
        # [优化点 3]: 更新调度器 (必须传入监控指标 avg_loss)
        scheduler.step(avg_loss)

        if (epoch + 1) % 10 == 0:
            # 在日志里打印一个当前的学习率
            current_lr = optimizer.param_groups[0]['lr']
            IDKN_logger.info(f'Epoch: {epoch + 1:04d}, Loss: {avg_loss:.6f}, LR: {current_lr:.6f}, Time: {time.time() - t_total:.2f}s')

        IDKN_logger.info(f'Epoch: {epoch + 1:04d}, Loss: {avg_loss:.6f}, Time: {time.time() - t_total:.2f}s')

        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_epoch = epoch
            bad_counter = 0
            save_model(model, optimizer, epoch, CHECKPOINTS_PATH)
        else:
            bad_counter += 1

        if bad_counter >= 150:
            IDKN_logger.info("Early stopping.")
            break

    IDKN_logger.info("Optimization Finished!")
    IDKN_logger.info("Total time elapsed: {:.2f}s".format(time.time() - t_total))
    IDKN_logger.info("Best Epoch: {:04d} with loss: {:.4f}".format(best_epoch, best_loss))
    IDKN_logger.info("Best model saved in: {}".format(CHECKPOINTS_PATH))
