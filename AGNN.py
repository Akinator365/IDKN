import os
import random
import time

import numpy as np
import torch
from torch_geometric.graphgym import optim

from Model import GAE


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


# def compute_test():
#     model.eval()
#     output = model(node_feature, adj)
#     loss_test = torch.nn.functional.mse_loss(output, label_t)
#
#     print("Test set results:",
#           "loss= {}".format(loss_test.data.item()),
#           )


if __name__ == '__main__':

    TRAIN_ADJ_PATH = os.path.join(os.getcwd(), 'data', 'adj', 'train')
    ba_path = os.path.join(TRAIN_ADJ_PATH, 'BA_graph', 'BA_500_3', 'BA_500_3_0_adj.npy')

    adj_BA = np.load(ba_path)
    adj_BA = torch.FloatTensor(adj_BA)

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
        if loss_values[-1] < best:
            best = loss_values[-1]
            best_epoch = epoch
            bad_counter = 0
        else:
            bad_counter += 1

        if bad_counter == 100:
            bad_counter += 1
    print("Optimization Finished!")
    print("Total time elapsed: {}s".format(time.time() - t_total))
    # compute_test()
    model.eval()
    node_feature_BA, A = model(torch.tensor(np.identity(adj_BA.shape[0])).float(), adj_BA)