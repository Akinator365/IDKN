import os

import numpy as np
import torch
from scipy.stats import kendalltau

from AGNN_Prediction import load_model
from Model import CGNN
from Utils import pickle_read

if __name__ == '__main__':

    TRAIN_ADJ_PATH = os.path.join(os.getcwd(), 'data', 'adj', 'train')
    TRAIN_DATASET_PATH = os.path.join(os.getcwd(), 'data', 'networks', 'train')
    ba_path = os.path.join(TRAIN_ADJ_PATH, 'BA_graph', 'BA_500_3', 'BA_500_3_0_adj.npy')
    TRAIN_LABELS_PATH = os.path.join(os.getcwd(), 'data', 'labels', 'train')
    label_path = os.path.join(TRAIN_LABELS_PATH, 'BA_graph', 'BA_500_3', 'BA_500_3_0_labels.npy')

    lable_BA = np.load("./lable_BA_1000_4.npy")
    lable_BA_t = torch.tensor(lable_BA).float()

    adj_BA = pickle_read(ba_path)
    adj_BA = torch.FloatTensor(adj_BA)

    # 保存路径设置
    best_embedding_path = os.path.join(os.getcwd(), 'best_embedding.npy')
    best_loss_path = os.path.join(os.getcwd(), 'best_loss.txt')

    node_feature = np.load(best_embedding_path)
    # 转换为 PyTorch 张量
    node_feature = torch.FloatTensor(node_feature)
    label = np.load(label_path)
    label_t =  torch.tensor(label).float()

    # 加载模型检查点
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    best = 2999
    checkpoint_path = f"./training/IDKN/2025-01-13_10-57-50/checkpoint_{best}_epoch.pkl"

    model = load_model(checkpoint_path, CGNN, device)

    model.eval()
    output = model(node_feature, adj_BA)

    # model_cgnn = CGNN()
    # model_cgnn.eval()
    # output = model_cgnn(node_feature, adj_BA)

    print(kendalltau(output.detach().numpy(), label_t))