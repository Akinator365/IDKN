import datetime
import json
import os
import random
import time

import numpy as np
import torch
from scipy.stats import kendalltau
from torch_geometric.graphgym import optim
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import dense_to_sparse

from Model import CGNN
from Utils import pickle_read

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

    cumsums = torch.cumsum(preds_sorted_by_true_minus_max.exp().flip(dims=[1]), dim=1).flip(dims=[1])

    observation_loss = torch.log(cumsums + eps) - preds_sorted_by_true_minus_max
    # print('listmle obs loss:', observation_loss)

    observation_loss[mask] = 0.0
    listmle = torch.mean(torch.sum(observation_loss, dim=1))
    #print('listmle loss:', listmle.item())

    return listmle


def train(epoch, node_feature, adj, label_t):
    t = time.time()
    model.train()

    output = model(node_feature, adj)
    loss_train = listMLE(output, label_t)
    # loss_train = torch.nn.functional.mse_loss(output, label_t)
    optimizer.zero_grad()
    loss_train.backward()
    optimizer.step()
    model.eval()
    output = model(node_feature, adj)

    loss_val = torch.nn.functional.mse_loss(output, label_t)

    print('Epoch: {:04d}'.format(epoch + 1),
          'loss_train: {}'.format(loss_train.data.item()),

          'loss_val: {}'.format(loss_val.data.item()),

          'time: {}s'.format(time.time() - t))
    return loss_train.data.item()

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

    # 从文件中读取参数
    with open("Network_Parameters_small.json", "r") as f:
        network_params = json.load(f)

    data_list = []  # 用于存储多个图的数据
    print("Processing graphs...")
    for network in network_params:
        network_type = network_params[network]['type']
        num_graph = network_params[network]['num']
        print(f'Processing {network} graphs...')
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
            data = Data(x=x, edge_index=edge_index, num_nodes=num_nodes, adj=adj, y=y)

            #data = Data(x=x, edge_index=edge_index, num_nodes=num_nodes, adj=adj, y=y)

            # 将图数据添加到 data_list 中
            data_list.append(data)

    # 将data_list分为训练集和测试集
    # 打乱数据
    np.random.shuffle(data_list)
    train_dataset = data_list[:round(len(data_list) * 0.8)]
    test_dataset = data_list[round(len(data_list) * 0.8):]
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    random.seed(17)
    np.random.seed(17)
    torch.manual_seed(17)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = CGNN().to(device)
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
            out = model(data.x, data.adj)  # Forward pass
            loss = listMLE(out, data.y) # Compute loss
            loss_val = torch.nn.functional.mse_loss(out, data.y)

            optimizer.zero_grad()  # Clear gradients
            loss.backward()  # Backward pass
            optimizer.step()  # Update parameters

            total_loss += loss.item()
            total_loss_val += loss_val.item()

        # Calculate average loss for the epoch
        avg_loss = total_loss / len(train_loader)
        avg_loss_val = total_loss_val / len(train_loader)

        print('Epoch: {:04d}'.format(epoch + 1),
              'loss_train: {}'.format(avg_loss),

              'loss_val: {}'.format(avg_loss_val),

              'time: {}s'.format(time.time() - t_total))

        if avg_loss < best:
            best = avg_loss
            best_epoch = epoch
            bad_counter = 0
            save_model(epoch, CHECKPOINTS_PATH)
        else:
            bad_counter += 1

        if bad_counter == 50:
            print("Early stopping triggered.")
            break
    print("Optimization Finished!")
    print("Total time elapsed: {}s".format(time.time() - t_total))
    print("Best Epoch: {:04d} with loss: {:.4f}".format(best_epoch, best))
    print("Best model saved in: ", CHECKPOINTS_PATH)
