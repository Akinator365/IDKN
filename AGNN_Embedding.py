import json
import os
import random
import time

import numpy as np
import torch
from torch.nn.functional import embedding
from torch_geometric.graphgym import optim

from Model import GAE
from Utils import pickle_read, normalize_adj_1


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def train(epoch, adj):
    t = time.time()
    model.train()

    x = torch.tensor(np.identity(adj.shape[0]), dtype=torch.float).to(device)
    adj = adj.to(device)  # 确保 adj 在 GPU

    x, A = model(x, adj)

    # loss_train = torch.norm(A - adj.sum(dim=1).reshape(-1, 1), p='fro').to(device)
    loss_train = torch.nn.functional.mse_loss(A, adj.sum(dim=1, keepdim=True).to(device))

    optimizer.zero_grad()
    loss_train.backward()
    optimizer.step()

    # 计算验证损失 (避免梯度计算)
    model.eval()
    with torch.no_grad():
        x, A = model(torch.tensor(np.identity(adj.shape[0])).float().to(device), adj)
        # loss_val = torch.norm(A - adj.sum(dim=1).reshape(-1, 1), p='fro').to(device)
        loss_val = torch.nn.functional.mse_loss(A, adj.sum(dim=1, keepdim=True).to(device))

    print('Epoch: {:04d}'.format(epoch + 1),
          'loss_train: {}'.format(loss_train.data.item()),

          'loss_val: {}'.format(loss_val.data.item()),

          'time: {}s'.format(time.time() - t))
    return loss_train.data.item()


def GenerateEmbedding(EMBEDDING_PATH, ADJ_PATH, network_params):
    global model, optimizer
    for network in network_params:
        network_type = network_params[network]['type']
        num_graph = network_params[network]['num']
        print(f'Processing {network} graphs...')
        for id in range(num_graph):
            network_name = f"{network}_{id}"
            embedding_path = os.path.join(EMBEDDING_PATH, network_type + '_graph', network,
                                          network_name + "_embedding.npy")

            # 如果文件已经存在，则跳过
            if os.path.exists(embedding_path):
                print(f"File {embedding_path} already exists, skipping...")
                continue
            else:
                print(f"Processing {network_name}")

            adj_path = os.path.join(ADJ_PATH, network_type + '_graph', network, network_name + '_adj.npy')
            adj = pickle_read(adj_path)
            adj = torch.FloatTensor(adj)
            adj = normalize_adj_1(torch.FloatTensor(adj)).to(device)

            # 初始化变量
            best_node_feature = None
            best_loss = float('inf')

            model = GAE(adj.shape[0], 48).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

            t_total = time.time()
            loss_values = []
            bad_counter = 0
            best_epoch = 0

            for epoch in range(500):
                loss_values.append(train(epoch, adj))

                # 更新最优结果
                if loss_values[-1] < best_loss:
                    best_loss = loss_values[-1]
                    best_epoch = epoch
                    # 获取当前最优嵌入
                    model.eval()
                    with torch.no_grad():
                        best_node_feature, _ = model(torch.tensor(np.identity(adj.shape[0])).float().to(device), adj)

                    os.makedirs(os.path.dirname(embedding_path), exist_ok=True)
                    # 保存最优嵌入
                    # np.save(embedding_path, best_node_feature.detach().numpy())
                    np.save(embedding_path, best_node_feature.detach().cpu().numpy())

                    print(f"Best Loss: {best_loss}\nEpoch: {best_epoch}")
                    # 早停机制
                if epoch > 0 and loss_values[-1] >= loss_values[-2]:
                    bad_counter += 1
                else:
                    bad_counter = 0  # 只要损失下降，就重置计数器

                if bad_counter >= 50:
                    print(f"Early stopping at epoch {epoch + 1} due to no improvement.")
                    break

            print("Optimization Finished!")
            print("Total time elapsed: {}s".format(time.time() - t_total))

            # 输出最优结果
            print(f"Best Loss: {best_loss} at Epoch: {best_epoch}")
            print(f"Best embedding saved to: {embedding_path}")


if __name__ == '__main__':

    TRAIN_EMBEDDING_PATH = os.path.join(os.getcwd(), 'data', 'embedding', 'train')
    TEST_EMBEDDING_PATH = os.path.join(os.getcwd(), 'data', 'embedding', 'test')
    REALWORLD_EMBEDDING_PATH = os.path.join(os.getcwd(), 'data', 'embedding', 'realworld')
    TRAIN_ADJ_PATH = os.path.join(os.getcwd(), 'data', 'adj', 'train')
    TEST_ADJ_PATH = os.path.join(os.getcwd(), 'data', 'adj', 'test')
    REALWORLD_ADJ_PATH = os.path.join(os.getcwd(), 'data', 'adj', 'realworld')

    # Training setup
    random.seed(17)
    np.random.seed(17)
    torch.manual_seed(17)

    # 从文件中读取参数
    with open("Network_Parameters_middle.json", "r") as f:
        train_network_params = json.load(f)

    with open("Network_Parameters_test.json", "r") as f:
        test_network_params = json.load(f)

    GenerateEmbedding(TRAIN_EMBEDDING_PATH, TRAIN_ADJ_PATH, train_network_params)
    GenerateEmbedding(TEST_EMBEDDING_PATH, TEST_ADJ_PATH, test_network_params)

