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
from Model import CGNN_New, CGNN_GAT, GDN_SIR_Predictor
from Utils import pickle_read, get_logger, sparse_adj_to_edge_index

DEFAULT_EPS = 1e-10
PADDED_Y_VALUE = -1


def listMLE(y_pred, y_true, eps=DEFAULT_EPS, padded_value_indicator=PADDED_Y_VALUE):
    """
    ListMLE loss.
    注意：这里输入的 y_pred 和 y_true 应该是属于【同一张图】的节点。
    """
    y_pred = y_pred.view(1, -1)
    y_true = y_true.view(1, -1)

    random_indices = range(y_pred.shape[-1])
    y_pred_shuffled = y_pred[:, random_indices]
    y_true_shuffled = y_true[:, random_indices]

    y_true_sorted, indices = y_true_shuffled.sort(descending=True, dim=-1)
    mask = y_true_sorted == padded_value_indicator

    preds_sorted_by_true = torch.gather(y_pred_shuffled, dim=1, index=indices)
    preds_sorted_by_true[mask] = float("-inf")

    max_pred_values, _ = preds_sorted_by_true.max(dim=1, keepdim=True)
    preds_sorted_by_true_minus_max = preds_sorted_by_true - max_pred_values

    # 数值稳定性 Clip
    preds_sorted_by_true_minus_max = torch.clamp(preds_sorted_by_true_minus_max, min=-100, max=100)

    cumsums = torch.cumsum(preds_sorted_by_true_minus_max.exp().flip(dims=[1]), dim=1).flip(dims=[1])
    observation_loss = torch.log(cumsums + eps) - preds_sorted_by_true_minus_max

    observation_loss[mask] = 0.0
    return torch.mean(torch.sum(observation_loss, dim=1))

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
    model = GDN_SIR_Predictor(hidden_dim=64).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

    # 使用 ReduceLROnPlateau (智能调度)
    # mode='min': 监测指标越小越好 (Loss)
    # factor=0.5: 每次调整为原来的 0.5 倍
    # patience=50: 如果 50 个 epoch 内 Loss 没有明显下降(threshold)，就触发调整
    # min_lr=1e-6: 学习率下限
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.8,
        patience=80,
        verbose=True,
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

            # Forward
            out = model(data)  # out shape: [total_nodes_in_batch]

            # 按图拆分计算 Loss (Graph-wise Loss)
            # data.batch 是一个向量 [0,0,0, 1,1,1, ...] 标记每个节点属于 batch 中的哪张图
            batch_loss = 0
            num_graphs = data.batch.max().item() + 1

            # 对 Batch 中的每一张图单独计算 Ranking Loss
            for i in range(num_graphs):
                mask = (data.batch == i)
                if mask.sum() > 0:  # 确保图非空
                    pred_i = out[mask]
                    true_i = data.y[mask]
                    # 只在单张图内部进行排序比较
                    # 1. 计算该图的节点数
                    num_nodes_i = data.y[mask].size(0)

                    # 2. 计算 Loss 并除以节点数 (归一化)
                    # 这样 Loss 就变成了 "平均每个节点的排序误差"
                    # 数值会从 2600 变成 2.6 左右
                    loss_i = listMLE(pred_i, true_i) / num_nodes_i

                    batch_loss += loss_i

            # 取平均 Loss
            loss = batch_loss / num_graphs

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
