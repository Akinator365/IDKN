import datetime
import json
import os

import numpy as np
import torch
from scipy import stats
from sympy import false
from torch.cuda import graph
from torch.optim.lr_scheduler import StepLR
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.loader import DataLoader   # 替换导入路径
from torch_geometric.utils import dense_to_sparse, from_networkx

from Model import *
from Utils import *


def test(loader):
    model.eval()
    loss = 0
    rank = 0
    for data in loader:  # Iterate in batches over the training/test dataset.
        data = data.to(device)
        out = model(data.x, data.x, data.edge_index, data.num_nodes)
        loss += criterion(out, data.y.view(-1, 1))
        rank += kendall_rank_coffecient(out.cpu().detach().numpy(), data.y.view(-1, 1).cpu().detach().numpy(),
                                        data.num_graphs, data)

    return loss / len(loader.dataset), rank / len(loader.dataset)


def kendall_rank_coffecient(out, label, batch_size, data):
    sum = 0
    last_split = 0
    for i in range(batch_size):
        batch_node = data[i].num_nodes
        out_node = out[last_split: last_split + batch_node]
        label_node = label[last_split: last_split + batch_node]
        out_rank = np.argsort(out_node, axis=0).reshape(-1)
        label_rank = np.argsort(label_node, axis=0).reshape(-1)
        tau, p_value = stats.kendalltau(label_rank, out_rank)
        sum += tau
        last_split = last_split + batch_node

    return sum


def save_model(epoch):
    checkpoint = {"model_state_dict": model.state_dict(),
                  "optimizer_state_dict": optimizer.state_dict(),
                  "epoch": epoch}
    path_checkpoint = os.path.join(CHECKPOINTS_PATH, "checkpoint_{}_epoch.pkl".format(epoch))
    torch.save(checkpoint, path_checkpoint)


if __name__ == '__main__':

    TRAIN_LABELS_PATH = os.path.join(os.getcwd(), 'data', 'labels', 'train')
    TRAIN_FEATURES_PATH = os.path.join(os.getcwd(), 'data', 'features', 'train')
    TRAIN_ADJ_PATH = os.path.join(os.getcwd(), 'data', 'adj', 'train')
    TRAIN_ROLES_PATH = os.path.join(os.getcwd(), 'data', 'roles', 'train')
    adj_path = TRAIN_ADJ_PATH
    features_path = TRAIN_FEATURES_PATH
    labels_path = TRAIN_LABELS_PATH
    roles_path = TRAIN_ROLES_PATH

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
            single_features_path = os.path.join(features_path, network_type + '_graph', network, network_name + '_features.npy')
            single_labels_path = os.path.join(labels_path, network_type + '_graph', network, network_name + '_labels.npy')
            single_roles_path = os.path.join(roles_path, network_type + '_graph', network, network_name + '_roles.npy')

            adj_matrix = pickle_read(single_adj_path)
            node_features = np.load(single_features_path)
            labels = np.load(single_labels_path)
            roles = np.load(single_roles_path)

            # 假设 adj_matrix 是一个邻接矩阵，我们需要将其转为边索引格式
            edge_index = dense_to_sparse(torch.tensor(adj_matrix))[0]  # 转为 edge_index 格式

            # 对节点特征进行归一化
            node_features = min_max_normalization(node_features)  # 归一化操作
            # 对节点角色特征进行归一化
            roles = min_max_normalization(roles)  # 归一化操作

            # 目标长度
            target_length = 10

            # 当前矩阵的列数
            current_length = roles.shape[1]

            # 如果列数小于目标长度，进行填充
            if current_length < target_length:
                # 在原矩阵后面添加零列
                padding = np.zeros((roles.shape[0], target_length - current_length))
                roles_padded = np.hstack((roles, padding))
            else:
                # 如果列数大于等于目标长度，直接截断
                roles_padded = roles[:, :target_length]

            # 转换为 PyTorch 张量
            x1 = torch.tensor(node_features, dtype=torch.float)  # 节点特征
            x2 = torch.tensor(roles_padded, dtype=torch.float)  # 节点角色特征
            y = torch.tensor(labels, dtype=torch.float)  # 标签

            # 计算节点数
            num_nodes = x1.size(0)  # 或者根据 adj_matrix 计算节点数：num_nodes = adj_matrix.shape[0]

            # 确保 x2 是一个 PyTorch Tensor
            roles_padded_tensor = torch.tensor(roles_padded, dtype=torch.float)

            # 创建 PyTorch Geometric 的 Data 对象
            data = Data(x=x1, x2=roles_padded_tensor, edge_index=edge_index, num_nodes=num_nodes, y=y)

            # 将图数据添加到 data_list 中
            data_list.append(data)

    # 使用 DataLoader 加载数据集
    #train_loader = DataLoader(train_dataset, batch_size=5, shuffle=True)
    # 将data_list分为训练集和测试集
    # 打乱数据
    np.random.shuffle(data_list)
    train_dataset = data_list[:round(len(data_list) * 0.8)]
    test_dataset = data_list[round(len(data_list) * 0.8):]
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    print(f"Number of batches in train_loader: {len(train_loader)}")

    for batch in train_loader:
        if batch is None:
            print("Received an empty batch!")
        else:
            print(f"Batch: {batch}")

    # 迭代 DataLoader，检查每个批次的数据
    for batch in train_loader:
        print(f"Batch:")
        print(f"Node features (x1): {batch.x.shape}")  # 打印节点特征的形状
        print(f"Edge index (edge_index): {batch.edge_index.shape}")  # 打印边索引的形状
        print(f"Labels (y): {batch.y.shape}")  # 打印标签的形状

        # 打印每个批次的部分数据
        print(f"First graph node features (x1[0]): {batch.x[0]}")  # 打印第一个图的节点特征
        print(f"First graph edge indices (edge_index[:, 0]): {batch.edge_index[:, 0]}")  # 打印第一个图的边索引
        print(f"First graph label (y[0]): {batch.y[0]}")  # 打印第一个图的标签

        break  # 只打印一个批次的数据，避免输出过多


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #model = IDKN().to(device)
    #model = IDKN_cat().to(device)
    model = IDKN_Attention().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    criterion = torch.nn.MSELoss(reduction='mean')
    scheduler_1 = StepLR(optimizer, step_size=50, gamma=0.3)
    epoch_num = 200
    checkpoint_interval = 5
    date = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    CHECKPOINTS_PATH = os.path.join(os.getcwd(), 'training', 'IDKN', date)
    os.makedirs(CHECKPOINTS_PATH, exist_ok=True)
    LOGGING_PATH = CHECKPOINTS_PATH + '/Logs/'
    os.makedirs(LOGGING_PATH, exist_ok=True)
    IDKN_logger = get_logger(LOGGING_PATH + 'train.log')
    os.makedirs(CHECKPOINTS_PATH, exist_ok=True)
    patience_period = 10
    BEST_VAL_LOSS = 100
    BEST_TRA_LOSS = 100
    BEST_VAL_KEN = 0
    BEST_TRA_KEN = 0
    BEST_TRA_MODEL_EPOCH = 0
    BEST_VAL_MODEL_EPOCH = 0

    PATIENCE_CNT = 0

    for epoch in range(epoch_num):
        model.train()
        for data in train_loader:  # Iterate in batches over the training dataset.
            data = data.to(device)
            out = model(data.x, data.x2, data.edge_index, data.num_nodes)
            loss = criterion(out, data.y.view(-1, 1))

            #print(loss)
            #print(out.shape)
            # print(data.y.view(-1,1).shape)
            # print(data.x.shape)
            loss.backward()  # Derive gradients.
            optimizer.step()  # Update parameters based on gradients.
            optimizer.zero_grad()  # Clear gradients
        loss1 = test(train_loader)
        loss2 = test(test_loader)
        if loss2[0] < BEST_VAL_LOSS or loss1[0] < BEST_TRA_LOSS or loss2[1] > BEST_VAL_KEN or loss1[1] > BEST_TRA_KEN:
            if loss2[0] < BEST_VAL_LOSS:
                BEST_VAL_MODEL_EPOCH = epoch
            elif loss1[0] < BEST_TRA_LOSS:
                BEST_TRA_MODEL_EPOCH = epoch
            elif loss2[1] > BEST_VAL_KEN:
                IDKN_logger.info(
                    'Better_VAL_Kendall_EPOCH:[{}/{}] \t Kendal:[{}/{}]'.format(
                        epoch, epoch_num, loss2[1], BEST_VAL_KEN))
                BEST_VAL_KEN = loss2[1]
            elif loss1[1] > BEST_TRA_KEN:
                IDKN_logger.info(
                    'Better_TAR_Kendall_EPOCH:[{}/{}] \t Kendal:[{}/{}]'.format(
                        epoch, epoch_num, loss1[1], BEST_TRA_KEN))
                BEST_TRA_KEN = loss1[1]

            BEST_VAL_LOSS = min(loss2[0], BEST_VAL_LOSS)  # keep track of the best validation accuracy so far
            BEST_TRA_LOSS = min(loss1[0], BEST_TRA_LOSS)
            PATIENCE_CNT = 0  # reset the counter every time we encounter new best accuracy
            save_model(epoch)
        else:
            PATIENCE_CNT += 1  # otherwise keep counting

        if PATIENCE_CNT >= patience_period:
            IDKN_logger.info(
                'BEST_TRA_MODEL_EPOCH:[{}/{}]\t BEST_VAL_MODEL_EPOCH:[{}/{}]\t TestLoss={:.6f}'.format(
                    BEST_TRA_MODEL_EPOCH, epoch_num, BEST_VAL_MODEL_EPOCH, epoch_num,
                    BEST_VAL_LOSS))
            raise Exception('Stopping the training, the universe has no more patience for this training.')

        IDKN_logger.info(
            'Epoch:[{}/{}]\t TrainLoss={:.6f}\t TrainKendal={:.4f}\t TestLoss={:.6f}\t TestKendal={:.4f}'.format(epoch,
                                                                                                                 epoch_num,
                                                                                                                 loss1[
                                                                                                                     0],
                                                                                                                 loss1[
                                                                                                                     1],
                                                                                                                 loss2[
                                                                                                                     0],
                                                                                                                 loss2[
                                                                                                                     1]))
    IDKN_logger.info('finish training!')