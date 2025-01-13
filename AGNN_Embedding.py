import os
import random
import time

import numpy as np
import torch
from torch_geometric.graphgym import optim

from Model import GAE
from Utils import pickle_read


def train(epoch):
    t = time.time()
    model.train()

    x, A = model(torch.tensor(np.identity(adj_BA.shape[0])).float(), adj_BA)

    loss_train = torch.norm(A - adj_BA.sum(dim=1).reshape(-1, 1), p='fro')
    optimizer.zero_grad()
    loss_train.backward()
    optimizer.step()
    model.eval()
    x, A = model(torch.tensor(np.identity(adj_BA.shape[0])).float(), adj_BA)

    loss_val = torch.norm(A - adj_BA.sum(dim=1).reshape(-1, 1), p='fro')

    print('Epoch: {:04d}'.format(epoch + 1),
          'loss_train: {}'.format(loss_train.data.item()),

          'loss_val: {}'.format(loss_val.data.item()),

          'time: {}s'.format(time.time() - t))
    return loss_train.data.item()


if __name__ == '__main__':

    TRAIN_ADJ_PATH = os.path.join(os.getcwd(), 'data', 'adj', 'train')
    ba_path = os.path.join(TRAIN_ADJ_PATH, 'BA_graph', 'BA_500_3', 'BA_500_3_0_adj.npy')

    adj_BA = pickle_read(ba_path)
    adj_BA = torch.FloatTensor(adj_BA)

    # 保存路径设置
    best_embedding_path = os.path.join(os.getcwd(), 'best_embedding.npy')
    best_loss_path = os.path.join(os.getcwd(), 'best_loss.txt')

    # 初始化变量
    best_node_feature = None
    best_loss = float('inf')

    # Training setup
    random.seed(17)
    np.random.seed(17)
    torch.manual_seed(17)

    model = GAE(adj_BA.shape[0], 48)
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    t_total = time.time()
    loss_values = []
    bad_counter = 0
    best = 1000 + 1
    best_epoch = 0


    for epoch in range(500):
        loss_values.append(train(epoch))

        # 更新最优结果
        if loss_values[-1] < best_loss:
            best_loss = loss_values[-1]
            best_epoch = epoch
            # 获取当前最优嵌入
            model.eval()
            best_node_feature, _ = model(torch.tensor(np.identity(adj_BA.shape[0])).float(), adj_BA)

            # 保存最优嵌入
            np.save(best_embedding_path, best_node_feature.detach().numpy())
            with open(best_loss_path, 'w') as f:
                f.write(f"Best Loss: {best_loss}\nEpoch: {best_epoch}")

        if bad_counter == 100:
            break

    print("Optimization Finished!")
    print("Total time elapsed: {}s".format(time.time() - t_total))

    # 输出最优结果
    print(f"Best Loss: {best_loss} at Epoch: {best_epoch}")
    print(f"Best embedding saved to: {best_embedding_path}")

    model.eval()
    node_feature_BA, A = model(torch.tensor(np.identity(adj_BA.shape[0])).float(), adj_BA)
