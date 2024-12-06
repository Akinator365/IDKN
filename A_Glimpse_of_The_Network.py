import os
import warnings

import networkx as nx
from matplotlib import pyplot as plt

if __name__ == '__main__':
    SYNTHETIC_DATASET_PATH = os.path.join(os.getcwd(), 'data', 'train')
    REALWORLD_DATASET_PATH = os.path.join(os.getcwd(), 'data', 'realworld')
    Network = 'karate_club_graph'
    # Network = 'DNCEmails'
    GRAPH_PATH = os.path.join(REALWORLD_DATASET_PATH, Network + ".txt")


    # 从文件加载图
    G = nx.read_edgelist(GRAPH_PATH)
    # plot graph
    plt.figure()
    with warnings.catch_warnings():
        # catch matplotlib deprecation warning
        warnings.simplefilter('ignore')
        nx.draw(
            G,
            pos=nx.spring_layout(G, seed=42),
            with_labels=True,
        )
    plt.show()