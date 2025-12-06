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

# DEFAULT_EPS = 1e-10
DEFAULT_EPS = 0.0001
PADDED_Y_VALUE = -1
PADDED_INDEX_VALUE = -1


def listMLE(y_pred, y_true, eps=DEFAULT_EPS, padded_value_indicator=PADDED_Y_VALUE):
    """
    FROM: https://github.com/allegro/allRank/blob/master/allrank/models/losses/listMLE.py
    ListMLE loss introduced in "Listwise Approach to Learning to Rank - Theory and Algorithm".
    :param y_pred: predictions from the model, shape [batch_size, slate_length]
    :param y_true: ground truth labels, shape [batch_size, slate_length]
    :param eps: epsilon value, used for numerical stability
    :param padded_value_indicator: an indicator of the y_true index containing a padded item, e.g. -1
    :return: loss value, a torch.Tensor
    """
    # Reshape the input
    # if len(y_true.size()) == 1:
    y_pred = y_pred.view(1, -1)
    y_true = y_true.view(1, -1)
    #print('listmle: y_true:', y_true.size(), 'y_pred', y_pred.size())
    # shuffle for randomised tie resolution
    random_indices = range(y_pred.shape[-1])
    #print(y_pred.shape[-1])
    y_pred_shuffled = y_pred[:, random_indices]
    y_true_shuffled = y_true[:, random_indices]

    y_true_sorted, indices = y_true_shuffled.sort(descending=True, dim=-1)

    mask = y_true_sorted == padded_value_indicator

    preds_sorted_by_true = torch.gather(y_pred_shuffled, dim=1, index=indices)
    preds_sorted_by_true[mask] = float("-inf")

    max_pred_values, _ = preds_sorted_by_true.max(dim=1, keepdim=True)

    preds_sorted_by_true_minus_max = preds_sorted_by_true - max_pred_values


    # 增加数值稳定性的检查

    if torch.isnan(y_pred).any() or torch.isinf(y_pred).any():

        IDKN_logger.info("预测值中存在NaN或Inf")

    # 增加eps值以提高数值稳定性

    DEFAULT_EPS = 1e-10  # 改为更小的值

    # 在计算exp之前进行clip操作

    preds_sorted_by_true_minus_max = torch.clamp(

        preds_sorted_by_true_minus_max, 

        min=-100,  # 防止exp操作溢出

        max=100

    )

    cumsums = torch.cumsum(preds_sorted_by_true_minus_max.exp().flip(dims=[1]), dim=1).flip(dims=[1])

    observation_loss = torch.log(cumsums + eps) - preds_sorted_by_true_minus_max
    # print('listmle obs loss:', observation_loss)

    observation_loss[mask] = 0.0
    listmle = torch.mean(torch.sum(observation_loss, dim=1))
    #print('listmle loss:', listmle.item())

    return listmle


def save_model(model, optimizer, epoch, path):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch
    }
    path_checkpoint = os.path.join(path, "checkpoint_{}_epoch.pkl".format(epoch))
    torch.save(checkpoint, path_checkpoint)


def load_model(checkpoint_path, model_class, device):
    """
    加载保存的模型检查点
    :param checkpoint_path: 模型文件的路径 (.pkl 文件)
    :param model_class: 定义的模型类，例如 IDKN
    :param device: 设备（'cuda' 或 'cpu'）
    :return: 加载权重的模型实例
    """
    # 初始化模型
    model = model_class().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)  # 加载检查点
    model.load_state_dict(checkpoint['model_state_dict'])  # 加载模型参数
    return model

def preprocess_features(embeddings):
    """对嵌入向量进行预处理"""
    # 标准化
    mean = np.mean(embeddings, axis=0)
    std = np.std(embeddings, axis=0)
    normalized_embeddings = (embeddings - mean) / (std + 1e-8)
    
    # 确保没有极端值
    normalized_embeddings = np.clip(
        normalized_embeddings, 
        -5,  # 下限
        5    # 上限
    )
    
    return normalized_embeddings

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

    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    # 可以加一个 Scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.5)

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

            # Loss Calculation (ListMLE)
            # 注意：ListMLE 期望 [batch, seq_len]。
            # DataLoader 默认把多个图拼成一个大图 (Batch)。
            # 你的 ListMLE 实现中做了 view(1, -1)，这意味着它把整个 Batch (8个图的所有节点)
            # 当作一个长列表来排序。这在全局 ranking 上是可行的，但理论上最好是按图计算 loss 再平均。
            # 为了兼容你现有的 ListMLE，这里保持现状。
            loss = listMLE(out, data.y)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

        # Calculate average loss for the epoch
        avg_loss = total_loss / len(train_loader)
        scheduler.step()

        IDKN_logger.info(f'Epoch: {epoch + 1:04d}, Loss: {avg_loss:.6f}, Time: {time.time() - t_total:.2f}s')

        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_epoch = epoch
            bad_counter = 0
            save_model(model, optimizer, epoch, CHECKPOINTS_PATH)
        else:
            bad_counter += 1

        if bad_counter >= 100:
            IDKN_logger.info("Early stopping.")
            break

    IDKN_logger.info("Optimization Finished!")
    IDKN_logger.info("Total time elapsed: {:.2f}s".format(time.time() - t_total))
    IDKN_logger.info("Best Epoch: {:04d} with loss: {:.4f}".format(best_epoch, best_loss))
    IDKN_logger.info("Best model saved in: {}".format(CHECKPOINTS_PATH))
