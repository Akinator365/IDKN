import json
import os
import subprocess
import numpy as np
import networkx as nx


def run_struc2vec(input_path, output_path, dimensions=128, walk_length=30, num_walks=300,
                  window_size=10, iterations=5, workers=4,
                  directed=False, weighted=False,
                  OPT1=True, OPT2=True, OPT3=True, until_layer=None):
    """
    调用 struc2vec 生成 .emb 文件
    """
    command = [
        "python", "struct2vec/main.py",
        "--input", input_path,
        "--output", output_path,
        "--dimensions", str(dimensions),
        "--walk-length", str(walk_length),
        "--num-walks", str(num_walks),
        "--window-size", str(window_size),
        "--iter", str(iterations),
        "--workers", str(workers),
    ]

    command += ["--directed" if directed else "--undirected"]
    command += ["--weighted" if weighted else "--unweighted"]

    if OPT1:
        command += ["--OPT1", "True"]
    if OPT2:
        command += ["--OPT2", "True"]
    if OPT3:
        command += ["--OPT3", "True"]
        if until_layer is not None:
            command += ["--until-layer", str(until_layer)]

    print(f"[Run] {os.path.basename(input_path)}")
    print(">>>", " ".join(command))
    subprocess.run(command)


def emb_to_npy(emb_path, graph_path, npy_path):
    G = nx.read_edgelist(graph_path, nodetype=int)
    node_list = list(G.nodes())

    with open(emb_path, 'r') as f:
        lines = f.readlines()[1:]

    emb_dict = {int(line.split()[0]): list(map(float, line.split()[1:])) for line in lines}
    embeddings = np.array([emb_dict[node] for node in node_list])

    np.save(npy_path, embeddings)
    print(f"[Save] Embedding saved to: {npy_path}\n")


def GenerateStruct2Vec(STR_PATH, EMB_PATH, DATASET_PATH, network_params):
    def GetVec(graph_path, str_path, emb_path, name):
        if os.path.exists(str_path):
            print(f"File {str_path} already exists, skipping...")
            return
        print(f"Processing {name}")
        # 你要处理的图名列表（不含后缀）
        os.makedirs(os.path.dirname(str_path), exist_ok=True)
        os.makedirs(os.path.dirname(emb_path), exist_ok=True)

        run_struc2vec(
            input_path=graph_path,
            output_path=emb_path,
            dimensions=128,
            walk_length=30,
            num_walks=300,
            window_size=10,
            iterations=5,
            workers=16,
            directed=False,
            weighted=False,
            OPT1=True,
            OPT2=True,
            OPT3=True,
            until_layer=6
        )

        emb_to_npy(emb_path, graph_path, str_path)
        print(f'pickle of {name} saved')

    for network in network_params:
        params = network_params[network]
        network_type = params['type']
        print(f'Processing {network} graphs...')

        entries = []
        if network_type == 'realworld':
            # Realworld 类型路径构造
            graph_path = os.path.join(DATASET_PATH, f"{network}.txt")
            str_path = os.path.join(STR_PATH, f"{network}_vec.npy")
            emb_path = os.path.join(EMB_PATH, f"{network}.emb")
            entries.append((graph_path, str_path, emb_path, network))
        else:
            # 合成数据集路径构造
            base_dir = f"{network_type}_graph"
            for id in range(params['num']):
                network_name = f"{network}_{id}"
                graph_path = os.path.join(DATASET_PATH, base_dir, network, f"{network_name}.txt")
                str_path = os.path.join(STR_PATH, base_dir, network, f"{network_name}_vec.npy")
                emb_path = os.path.join(EMB_PATH, base_dir, network, f"{network_name}.emb")
                entries.append((graph_path, str_path, emb_path, network_name))

        for graph_path, str_path, emb_path, name in entries:
            GetVec(graph_path, str_path, emb_path, name)


if __name__ == "__main__":
    TRAIN_DATASET_PATH = os.path.join(os.getcwd(), 'data', 'networks', 'train')
    TEST_DATASET_PATH = os.path.join(os.getcwd(), 'data', 'networks', 'test')
    REALWORLD_DATASET_PATH = os.path.join(os.getcwd(), 'data', 'networks', 'realworld')
    TRAIN_STR_PATH = os.path.join(os.getcwd(), 'data', 'struct', 'vec', 'train')
    TEST_STR_PATH = os.path.join(os.getcwd(), 'data', 'struct', 'vec', 'test')
    REALWORLD_STR_PATH = os.path.join(os.getcwd(), 'data', 'struct', 'vec', 'realworld')
    TRAIN_EMB_PATH = os.path.join(os.getcwd(), 'data', 'struct', 'emb', 'train')
    TEST_EMB_PATH = os.path.join(os.getcwd(), 'data', 'struct', 'emb', 'test')
    REALWORLD_EMB_PATH = os.path.join(os.getcwd(), 'data', 'struct', 'emb', 'realworld')


    # 从文件中读取参数
    with open("Network_Parameters_middle.json", "r") as f:
        train_network_params = json.load(f)

    with open("Network_Parameters_test.json", "r") as f:
        test_network_params = json.load(f)

    with open("Network_Parameters_realworld.json", "r") as f:
        realworld_network_params = json.load(f)

    GenerateStruct2Vec(TRAIN_STR_PATH, TRAIN_EMB_PATH, TRAIN_DATASET_PATH, train_network_params)
    GenerateStruct2Vec(TEST_STR_PATH, TEST_EMB_PATH, TEST_DATASET_PATH, test_network_params)
    GenerateStruct2Vec(REALWORLD_STR_PATH, REALWORLD_EMB_PATH, REALWORLD_DATASET_PATH, realworld_network_params)
