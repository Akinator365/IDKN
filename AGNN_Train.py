import datetime
import json
import os
import random
import sys
import time

import numpy as np
import torch
from scipy.stats import kendalltau
from torch_geometric.graphgym import optim
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import dense_to_sparse, to_dense_adj

from Model import CGNN, CGNN_New
from Utils import pickle_read, get_logger

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


def save_model(epoch, path):
    checkpoint = {"model_state_dict": model.state_dict(),
                  "optimizer_state_dict": optimizer.state_dict(),
                  "epoch": epoch}
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
    TRAIN_EMBEDDING_PATH = os.path.join(os.getcwd(), 'data', 'embedding', 'train')
    adj_path = TRAIN_ADJ_PATH
    labels_path = TRAIN_LABELS_PATH
    embedding_path = TRAIN_EMBEDDING_PATH

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
            single_adj_path = os.path.join(adj_path, network_type + '_graph', network, network_name + '_adj.npy')
            single_labels_path = os.path.join(labels_path, network_type + '_graph', network, network_name + '_labels.npy')
            single_embedding_path = os.path.join(embedding_path, network_type + '_graph', network, network_name + '_embedding.npy')

            node_feature = np.load(single_embedding_path)
            # 转换为 PyTorch 张量
            x = torch.FloatTensor(node_feature)

            adj_matrix = pickle_read(single_adj_path)
            adj = torch.FloatTensor(adj_matrix)
            labels = np.load(single_labels_path)

            # adj_matrix 是一个邻接矩阵，我们需要将其转为边索引格式
            edge_index = dense_to_sparse(torch.tensor(adj_matrix))[0]  # 转为 edge_index 格式

            # 计算节点数
            num_nodes = adj_matrix.shape[0] # 根据 adj_matrix 计算节点数：num_nodes = adj_matrix.shape[0]

            # 转换为 PyTorch 张量
            y = torch.tensor(labels, dtype=torch.float)  # 标签

            # 创建 PyTorch Geometric 的 Data 对象
            data = Data(x=x, edge_index=edge_index, y=y)

            #data = Data(x=x, edge_index=edge_index, num_nodes=num_nodes, adj=adj, y=y)

            # 将图数据添加到 data_list 中
            data_list.append(data)

    # 将data_list分为训练集和测试集
    # 打乱数据
    np.random.shuffle(data_list)
    # train_dataset = data_list[:round(len(data_list) * 0.8)]
    # test_dataset = data_list[round(len(data_list) * 0.8):]
    train_loader = DataLoader(data_list, batch_size=8, shuffle=True, follow_batch=['x'])
    # test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    random.seed(17)
    np.random.seed(17)
    torch.manual_seed(17)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    IDKN_logger.info(f"using device:{device}")

    model = CGNN_New().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)

    t_total = time.time()
    loss_values = []
    bad_counter = 0
    best = float('inf')  # 初始化为正无穷
    best_epoch = 0

    for epoch in range(3000):
        model.train()
        total_loss = 0
        total_loss_val = 0
        for data in train_loader:  # Batch training with DataLoader
            data = data.to(device)  # Move batch to GPU/CPU
            # 转回稠密邻接矩阵
            # adj_matrix_reconstructed = to_dense_adj(data.edge_index, max_num_nodes=data.num_nodes).squeeze(0)

            out = model(data.x, data.edge_index)  # Forward pass
            loss = listMLE(out, data.y) # Compute loss
            #loss = torch.nn.functional.mse_loss(out, data.y)
            loss_val = torch.nn.functional.mse_loss(out, data.y)

            optimizer.zero_grad()  # Clear gradients
            loss.backward()  # Backward pass
            
            # 添加梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()  # Update parameters

            total_loss += loss.item()
            total_loss_val += loss_val.item()

        # Calculate average loss for the epoch
        avg_loss = total_loss / len(train_loader)
        avg_loss_val = total_loss_val / len(train_loader)

        IDKN_logger.info('Epoch: {:04d}, loss_train: {:.6f}, loss_val: {:.6f}, time: {:.2f}s'.format(
            epoch + 1, avg_loss, avg_loss_val, time.time() - t_total
        ))

        if avg_loss < best:
            best = avg_loss
            best_epoch = epoch
            bad_counter = 0
            save_model(epoch, CHECKPOINTS_PATH)
        else:
            bad_counter += 1

        if bad_counter == 50:
            IDKN_logger.info("Early stopping triggered.")
            break
    IDKN_logger.info("Optimization Finished!")
    IDKN_logger.info("Total time elapsed: {:.2f}s".format(time.time() - t_total))
    IDKN_logger.info("Best Epoch: {:04d} with loss: {:.4f}".format(best_epoch, best))
    IDKN_logger.info("Best model saved in: {}".format(CHECKPOINTS_PATH))
