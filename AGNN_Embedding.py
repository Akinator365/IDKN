import json
import os
import random
import sys
import time

import numpy as np
import scipy as sp
import torch
from torch.nn.functional import embedding
from torch_geometric.graphgym import optim
from torch_geometric.utils import dense_to_sparse
import torch.nn.functional as F
from Model import TraditionalGAE, RevisedGAE, optimitzedGAE, learnableGAE, resGAE, res_rand_GAE, rand_GAE, \
    struct_start_GAE
from Utils import pickle_read, normalize_adj_1, sparse_adj_to_edge_index, get_logger

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def GenerateEmbedding(EMBEDDING_PATH, ADJ_PATH, VEC_PATH, network_params):
    # ===== 全局统计容器 =====
    weak_decrease_records = []  # 记录全程下降 <0.1 的任务名称

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

        def train_traditional(epoch, adj):
            model.train()
            adj = adj.to(device)

            # 前向传播获取嵌入
            z = model(adj)

            # 计算内积重构邻接矩阵
            adj_recon = torch.sigmoid(z @ z.T)  # [N, N]

            # 计算二元交叉熵损失
            pos_weight = (adj.numel() - adj.sum()) / adj.sum()  # 处理类别不平衡
            loss = F.binary_cross_entropy_with_logits(z @ z.T, adj, pos_weight=pos_weight)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 验证损失（可选）
            model.eval()
            with torch.no_grad():
                z_val = model(adj)
                adj_recon_val = torch.sigmoid(z_val @ z_val.T)
                loss_val = F.binary_cross_entropy_with_logits(z_val @ z_val.T, adj)

            print(f'Epoch: {epoch + 1:04d} | Loss: {loss.item():.4f} | Val Loss: {loss_val.item():.4f}')
            return loss.item()

        if os.path.exists(embedding_path):
            print(f"File {embedding_path} already exists, skipping...")
            return

        print(f"Processing {name}")
        adj_sparse = sp.sparse.load_npz(adj_path)  # 加载压缩稀疏矩阵
        adj = torch.FloatTensor(adj_sparse.toarray()).to(device) # 转换为密集矩阵
        embedding = np.load(vec_path)
        embeddings_tensor = torch.tensor(embedding, dtype=torch.float32).to(device)

        #model = optimitzedGAE(adj.shape[0]).to(device)
        # model = resGAE(adj.shape[0]).to(device)
        # model = res_rand_GAE(adj.shape[0]).to(device)
        # model = rand_GAE(adj.shape[0]).to(device)
        # model = learnableGAE(adj.shape[0]).to(device)
        #model = crazyGAE(adj.shape[0]).to(device)
        model = RevisedGAE(adj.shape[0]).to(device)
        # model = TraditionalGAE(adj.shape[0]).to(device)
        #model = struct_start_GAE(adj.shape[0], embeddings_tensor).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

        t_total = time.time()
        best_loss = float('inf')
        best_feature = None  # 存储最佳嵌入的变量
        bad_counter = 0

        # ===== 新增：初始化监控变量 =====
        initial_loss = None
        final_loss = None

        for epoch in range(2000):
            # loss = train_traditional(epoch, adj)
            loss = train(epoch, adj)

            # ===== 记录初始 loss（首个 epoch）=====
            if epoch == 3:
                initial_loss = loss
                print(f"初始 loss: {initial_loss:.4f}")

            # 更新最佳结果逻辑
            if loss < best_loss:
                print(f"New best loss: {loss:.4f} (improved from {best_loss:.4f})")
                best_loss = loss
                with torch.no_grad():
                    # ，_
                    node_features, _ = model(
                        adj
                    )
                    best_feature = node_features.detach().cpu().numpy()
                bad_counter = 0  # 重置计数器
            else:
                bad_counter += 1  # 累计未改进次数

            # 早停判断（连续100次未改进）
            if bad_counter >= 100:
                print(f"Early stopping: No improvement for {bad_counter} epochs")
                final_loss = loss  # 记录早停时的 loss
                break

        # ===== 处理完整训练未早停的情况 =====
        if final_loss is None:
            final_loss = loss  # 记录最后一个 epoch 的 loss

        # ===== 计算全程 loss 变化 =====
        delta_loss = initial_loss - final_loss
        if delta_loss < 0.2:  # 全程下降值小于阈值
            IDKN_logger.info(f"[警告] 全程 loss 下降值仅 {delta_loss:.2f}（初始 {initial_loss:.2f} → 最终 {final_loss:.2f}），图：{name}")
        # ===== 将异常情况记录到全局列表 =====
        nonlocal weak_decrease_records
        if delta_loss < 0.2:
            weak_decrease_records.append(name)

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

    # ===== 最终输出统计结果 =====
    IDKN_logger.info("\n===== 全局统计 =====")
    IDKN_logger.info(f"共有 {len(weak_decrease_records)} 个任务的全程 loss 下降值 <0.1")
    if len(weak_decrease_records) > 0:
        IDKN_logger.info("具体任务列表:", weak_decrease_records)


if __name__ == '__main__':

    TRAIN_EMBEDDING_PATH = os.path.join(os.getcwd(), 'data', 'embedding', 'train')
    TEST_EMBEDDING_PATH = os.path.join(os.getcwd(), 'data', 'embedding', 'test')
    REALWORLD_EMBEDDING_PATH = os.path.join(os.getcwd(), 'data', 'embedding', 'realworld')
    TRAIN_ADJ_PATH = os.path.join(os.getcwd(), 'data', 'adj', 'train')
    TEST_ADJ_PATH = os.path.join(os.getcwd(), 'data', 'adj', 'test')
    REALWORLD_ADJ_PATH = os.path.join(os.getcwd(), 'data', 'adj', 'realworld')
    TRAIN_VEC_PATH = os.path.join(os.getcwd(), 'data','struct', 'vec', 'train')
    TEST_VEC_PATH = os.path.join(os.getcwd(), 'data','struct', 'vec', 'test')
    REALWORLD_VEC_PATH = os.path.join(os.getcwd(), 'data','struct', 'vec', 'realworld')

    # 初始化日志记录器
    # 保证路径存在
    os.makedirs(TRAIN_EMBEDDING_PATH, exist_ok=True)
    LOGGING_PATH = os.path.join(TRAIN_EMBEDDING_PATH, 'train.log')
    # 传入 logger
    IDKN_logger = get_logger(LOGGING_PATH)
    # sys.stdout = IDKN_logger  # 让 print() 也写入日志

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
    #GenerateEmbedding(TEST_EMBEDDING_PATH, TEST_ADJ_PATH, TEST_VEC_PATH, test_network_params)
    #GenerateEmbedding(REALWORLD_EMBEDDING_PATH, REALWORLD_ADJ_PATH, REALWORLD_VEC_PATH, realworld_network_params)

