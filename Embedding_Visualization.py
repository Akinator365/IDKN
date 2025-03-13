import matplotlib.pyplot as plt
import scipy as sp
import torch
from sklearn.manifold import TSNE
import os
import json
import numpy as np
from Utils import pickle_read

def visualize_embeddings(embeddings, labels=None):
    # 使用t-SNE将高维嵌入降到2维
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    # 绘制散点图
    plt.figure(figsize=(10, 8))
    if labels is not None:
        scatter = plt.scatter(embeddings_2d[:, 0], 
                            embeddings_2d[:, 1], 
                            c=labels, 
                            cmap='rainbow')
        plt.colorbar(scatter)
    else:
        plt.scatter(embeddings_2d[:, 0], 
                   embeddings_2d[:, 1])
    
    plt.title('节点嵌入的t-SNE可视化')
    plt.xlabel('t-SNE维度1')
    plt.ylabel('t-SNE维度2')
    plt.show()
    

# main
if __name__ == '__main__':
    TRAIN_EMBEDDING_PATH = os.path.join(os.getcwd(), 'data', 'embedding', 'train')
    REALWORLD_EMBEDDING_PATH = os.path.join(os.getcwd(), 'data', 'embedding', 'realworld')
    TRAIN_ADJ_PATH = os.path.join(os.getcwd(), 'data', 'adj', 'train')
    REALWORLD_ADJ_PATH = os.path.join(os.getcwd(), 'data', 'adj', 'realworld')


    # 从文件中读取参数
    with open("Network_Parameters_small.json", "r") as f:
        network_params = json.load(f)

    #for network in network_params:
    network = 'BA_2000_3'
    network_type = network_params[network]['type']
    num_graph = network_params[network]['num']
    print(f'Processing {network} graphs...')
        #for id in range(num_graph):
    id = 2
    network_name = f"{network}_{id}"
    embedding_path = os.path.join(TRAIN_EMBEDDING_PATH, network_type + '_graph', network, network_name + "_embedding.npy")
    embeddings = np.load(embedding_path)
    adj_path = os.path.join(TRAIN_ADJ_PATH, network_type + '_graph', network, network_name + '_adj.npz')
    adj_sparse = sp.sparse.load_npz(adj_path)  # 加载压缩稀疏矩阵
    adj = torch.FloatTensor(adj_sparse.toarray())  # 转换为密集矩阵
    node_degrees = adj.sum(axis=1)

    visualize_embeddings(embeddings, node_degrees)


