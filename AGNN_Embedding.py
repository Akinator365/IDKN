import json
import os
import random
import time

import numpy as np
import scipy as sp
import torch
from torch.nn.functional import embedding
from torch_geometric.graphgym import optim
from torch_geometric.utils import dense_to_sparse
import torch.nn.functional as F
from Model import GAE, RevisedGAE, optimitzedGAE
from Utils import pickle_read, normalize_adj_1, sparse_adj_to_edge_index

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def GenerateEmbedding(EMBEDDING_PATH, ADJ_PATH, VEC_PATH, network_params):
    def GetEmbedding(name, adj_path, vec_path, embedding_path):
        # 定义训练函数在闭包内部以共享模型参数
        def train(epoch, adj):
            t = time.time()
            model.train()

            # x = torch.tensor(np.identity(adj.shape[0]), dtype=torch.float).to(device)
            adj = adj.to(device)  # 确保 adj 在 GPU

            x, A = model(adj)

            # loss_train = torch.norm(A - adj.sum(dim=1).reshape(-1, 1), p='fro').to(device)
            loss_train = torch.nn.functional.mse_loss(A, adj.sum(dim=1, keepdim=True).to(device))

            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()

            # 计算验证损失 (避免梯度计算)
            model.eval()
            with torch.no_grad():
                x, A = model(adj)
                # loss_val = torch.norm(A - adj.sum(dim=1).reshape(-1, 1), p='fro').to(device)
                loss_val = torch.nn.functional.mse_loss(A, adj.sum(dim=1, keepdim=True).to(device))

            print('Epoch: {:04d}'.format(epoch + 1),
                  'loss_train: {}'.format(loss_train.data.item()),

                  'loss_val: {}'.format(loss_val.data.item()),

                  'time: {}s'.format(time.time() - t))
            return loss_train.data.item()
        # 定义训练函数在闭包内部以共享模型参数
        def train_vec(epoch, adj, embedding):
            t = time.time()
            model.train()
            embedding = F.normalize(embedding.to(device), p=2, dim=1)

            # x = torch.tensor(np.identity(adj.shape[0]), dtype=torch.float).to(device)
            adj = adj.to(device)  # 确保 adj 在 GPU

            x, A = model(adj)

            # loss_train = torch.norm(A - adj.sum(dim=1).reshape(-1, 1), p='fro').to(device)
            # loss_train = torch.nn.functional.mse_loss(A, adj.sum(dim=1, keepdim=True).to(device))
            loss_train = 1 - F.cosine_similarity(A, embedding).mean()  # 平均余弦相似度损失

            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()

            # 计算验证损失 (避免梯度计算)
            model.eval()
            with torch.no_grad():
                x, A = model(adj)
                # loss_val = torch.norm(A - adj.sum(dim=1).reshape(-1, 1), p='fro').to(device)
                loss_val = torch.nn.functional.mse_loss(A, adj.sum(dim=1, keepdim=True).to(device))

            print('Epoch: {:04d}'.format(epoch + 1),
                  'loss_train: {}'.format(loss_train.data.item()),

                  'loss_val: {}'.format(loss_val.data.item()),

                  'time: {}s'.format(time.time() - t))
            return loss_train.data.item()
        if os.path.exists(embedding_path):
            print(f"File {embedding_path} already exists, skipping...")
            return

        print(f"Processing {name}")
        adj_sparse = sp.sparse.load_npz(adj_path)  # 加载压缩稀疏矩阵
        adj = torch.FloatTensor(adj_sparse.toarray()).to(device) # 转换为密集矩阵
        #embedding = np.load(vec_path)
        #embeddings_tensor = torch.tensor(embedding, dtype=torch.float32).to(device)

        # model = optimitzedGAE(adj.shape[0]).to(device)
        model = RevisedGAE(adj.shape[0]).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

        t_total = time.time()
        best_loss = float('inf')
        best_feature = None  # 存储最佳嵌入的变量
        bad_counter = 0

        for epoch in range(1000):
            # loss = train_vec(epoch, adj, embeddings_tensor)
            loss = train(epoch, adj)


            # 更新最佳结果逻辑
            if loss < best_loss:
                print(f"New best loss: {loss:.4f} (improved from {best_loss:.4f})")
                best_loss = loss
                with torch.no_grad():
                    node_features, _ = model(
                        adj
                    )
                    best_feature = node_features.detach().cpu().numpy()
                bad_counter = 0  # 重置计数器
            else:
                bad_counter += 1  # 累计未改进次数

            # 早停判断（连续50次未改进）
            if bad_counter >= 50:
                print(f"Early stopping: No improvement for {bad_counter} epochs")
                break

        # 训练结束后统一保存
        if best_feature is not None:
            os.makedirs(os.path.dirname(embedding_path), exist_ok=True)
            np.save(embedding_path, best_feature)
            print(f"Best embedding saved to: {embedding_path}")
        else:
            print("Warning: No valid embedding generated")

        print(f"Optimization Finished! Total time: {time.time() - t_total:.1f}s")

    for network in network_params:
        params = network_params[network]
        network_type = params['type']
        print(f'Processing {network} graphs...')

        entries = []
        if network_type == 'realworld':
            # Realworld 路径构造
            adj_path = os.path.join(ADJ_PATH, f"{network}_adj.npz")
            embedding_path = os.path.join(EMBEDDING_PATH, f"{network}_embedding.npy")
            vec_path  = os.path.join(VEC_PATH, f"{network}_vec.npy")
            entries.append((network, adj_path, vec_path, embedding_path))
        else:
            # 合成数据集路径构造
            base_dir = f"{network_type}_graph"
            for id in range(params['num']):
                name = f"{network}_{id}"
                adj_path = os.path.join(ADJ_PATH, base_dir, network, f"{name}_adj.npz")
                embedding_path = os.path.join(EMBEDDING_PATH, base_dir, network, f"{name}_embedding.npy")
                vec_path = os.path.join(VEC_PATH, base_dir, network, f"{name}_vec.npy")
                entries.append((name, adj_path, vec_path, embedding_path))

        for name, adj_path, vec_path, embedding_path in entries:
            GetEmbedding(name, adj_path, vec_path, embedding_path)


if __name__ == '__main__':

    TRAIN_EMBEDDING_PATH = os.path.join(os.getcwd(), 'data', 'embedding', 'train')
    TEST_EMBEDDING_PATH = os.path.join(os.getcwd(), 'data', 'embedding', 'test')
    REALWORLD_EMBEDDING_PATH = os.path.join(os.getcwd(), 'data', 'embedding', 'realworld')
    TRAIN_ADJ_PATH = os.path.join(os.getcwd(), 'data', 'adj', 'train')
    TEST_ADJ_PATH = os.path.join(os.getcwd(), 'data', 'adj', 'test')
    REALWORLD_ADJ_PATH = os.path.join(os.getcwd(), 'data', 'adj', 'realworld')
    TRAIN_VEC_PATH = os.path.join(os.getcwd(), 'data', 'vec', 'train')
    TEST_VEC_PATH = os.path.join(os.getcwd(), 'data', 'vec', 'test')
    REALWORLD_VEC_PATH = os.path.join(os.getcwd(), 'data', 'vec', 'realworld')
    # Training setup
    random.seed(17)
    np.random.seed(17)
    torch.manual_seed(17)

    # 从文件中读取参数
    with open("Network_Parameters_small.json", "r") as f:
        train_network_params = json.load(f)

    with open("Network_Parameters_test.json", "r") as f:
        test_network_params = json.load(f)

    with open("Network_Parameters_realworld.json", "r") as f:
        realworld_network_params = json.load(f)

    GenerateEmbedding(TRAIN_EMBEDDING_PATH, TRAIN_ADJ_PATH, TRAIN_VEC_PATH, train_network_params)
    GenerateEmbedding(TEST_EMBEDDING_PATH, TEST_ADJ_PATH, TEST_VEC_PATH, test_network_params)
    GenerateEmbedding(REALWORLD_EMBEDDING_PATH, REALWORLD_ADJ_PATH, REALWORLD_VEC_PATH, realworld_network_params)

