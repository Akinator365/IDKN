import json
import os

import numpy as np
import torch
from scipy.stats import kendalltau

from AGNN_Prediction import load_model
from Model import CGNN
from Utils import pickle_read, check_embeddings

if __name__ == '__main__':

    TRAIN_ADJ_PATH = os.path.join(os.getcwd(), 'data', 'adj', 'train')
    TRAIN_EMBEDDING_PATH = os.path.join(os.getcwd(), 'data', 'embedding', 'train')
    TRAIN_LABELS_PATH = os.path.join(os.getcwd(), 'data', 'labels', 'train')

    # 加载模型检查点
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    best = 260
    checkpoint_path = f"./training/IDKN/2025-03-03_21-29-47/checkpoint_{best}_epoch.pkl"

    model = load_model(checkpoint_path, CGNN, device)

    model.eval()

    # 从文件中读取参数
    with open("Network_Parameters_small.json", "r") as f:
        network_params = json.load(f)

    print("Processing graphs...")
    for network in network_params:
        network_type = network_params[network]['type']
        num_graph = network_params[network]['num']
        print(f'Processing {network} graphs...')
        for id in range(num_graph):
            network_name = f"{network}_{id}"
            adj_path = os.path.join(TRAIN_ADJ_PATH, network_type + '_graph', network, f'{network_name}_adj.npy')
            label_path = os.path.join(TRAIN_LABELS_PATH, network_type + '_graph', network,
                                      f'{network_name}_labels.npy')
            embedding_path = os.path.join(TRAIN_EMBEDDING_PATH, network_type + '_graph', network,
                                          f'{network_name}_embedding.npy')

            adj_BA = pickle_read(adj_path)
            adj_BA = torch.FloatTensor(adj_BA)

            node_feature = np.load(embedding_path)
            # check_embeddings(node_feature)
            
            # 转换为 PyTorch 张量
            node_feature = torch.FloatTensor(node_feature)
            label = np.load(label_path)
            label_t = torch.tensor(label).float()

            output = model(node_feature, adj_BA)

            print(kendalltau(output.detach().numpy(), label_t))

