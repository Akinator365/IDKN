import json
import os
import networkx as nx
import scipy
from Utils import *
from node2vec import Node2Vec

def GenerateNode2Vec(VEC_PATH, DATASET_PATH, network_params):
    def GetVec(graph_path, vec_path, name):
        if os.path.exists(vec_path):
            print(f"File {vec_path} already exists, skipping...")
            return
        print(f"Processing {name}")
        G = nx.read_edgelist(graph_path)
        print(G.number_of_nodes(), G.number_of_edges())

        # Node2Vec参数设置
        node2vec = Node2Vec(G, dimensions=128, walk_length=30, num_walks=300, workers=12)
        model = node2vec.fit(window=10, min_count=1)

        # 提取嵌入并保存
        embeddings = np.array([model.wv[str(i)] for i in range(len(G))])
        os.makedirs(os.path.dirname(vec_path), exist_ok=True)
        np.save(vec_path, embeddings)
        print(f'pickle of {name} saved')

    for network in network_params:
        params = network_params[network]
        network_type = params['type']
        print(f'Processing {network} graphs...')

        entries = []
        if network_type == 'realworld':
            # Realworld 类型路径构造
            graph_path = os.path.join(DATASET_PATH, f"{network}.txt")
            vec_path = os.path.join(VEC_PATH, f"{network}_vec.npy")
            entries.append((graph_path, vec_path, network))
        else:
            # 合成数据集路径构造
            base_dir = f"{network_type}_graph"
            for id in range(params['num']):
                network_name = f"{network}_{id}"
                graph_path = os.path.join(DATASET_PATH, base_dir, network, f"{network_name}.txt")
                vec_path = os.path.join(VEC_PATH, base_dir, network, f"{network_name}_vec.npy")
                entries.append((graph_path, vec_path, network_name))

        for graph_path, vec_path, name in entries:
            GetVec(graph_path, vec_path, name)


if __name__ == '__main__':
    TRAIN_DATASET_PATH = os.path.join(os.getcwd(), 'data', 'networks', 'train')
    TEST_DATASET_PATH = os.path.join(os.getcwd(), 'data', 'networks', 'test')
    REALWORLD_DATASET_PATH = os.path.join(os.getcwd(), 'data', 'networks', 'realworld')
    TRAIN_VEC_PATH = os.path.join(os.getcwd(), 'data', 'vec', 'train')
    TEST_VEC_PATH = os.path.join(os.getcwd(), 'data', 'vec', 'test')
    REALWORLD_VEC_PATH = os.path.join(os.getcwd(), 'data', 'vec', 'realworld')

    # 从文件中读取参数
    with open("Network_Parameters_small.json", "r") as f:
        train_network_params = json.load(f)

    with open("Network_Parameters_test.json", "r") as f:
        test_network_params = json.load(f)

    with open("Network_Parameters_realworld.json", "r") as f:
        realworld_network_params = json.load(f)

    GenerateNode2Vec(TRAIN_VEC_PATH, TRAIN_DATASET_PATH, train_network_params)
    GenerateNode2Vec(TEST_VEC_PATH, TEST_DATASET_PATH, test_network_params)
    GenerateNode2Vec(REALWORLD_VEC_PATH, REALWORLD_DATASET_PATH, realworld_network_params)