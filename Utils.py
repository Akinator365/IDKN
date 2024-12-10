import pickle
import logging



def pickle_read(path):
    with open(path, 'rb') as file:
        data = pickle.load(file)
    return data

def pickle_save(path, data):
    with open(path, 'wb') as file:
        pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


# 定义一个函数来读取文件并按第二列降序排列
def read_and_sort_txt(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.split()
            index = int(parts[0])  # 第一列
            value = float(parts[1])  # 第二列
            data.append((index, value))

    # 按第二列的值进行降序排列
    sorted_data = sorted(data, key=lambda x: x[1], reverse=True)
    sorted_indexes = [x[0] for x in sorted_data]

    return sorted_indexes

def read_edges(file_path):
    adj_list = {}

    # 打开并读取文件
    with open(file_path, 'r') as f:
        for line in f:
            # 每行是一个边，格式为 "node1 node2"
            node1, node2 = map(int, line.strip().split())

            # 如果节点1不在字典中，初始化它
            if node1 not in adj_list:
                adj_list[node1] = set()
            # 如果节点2不在字典中，初始化它
            if node2 not in adj_list:
                adj_list[node2] = set()

            # 将每条边加到两个节点的邻接表中
            adj_list[node1].add(node2)
            adj_list[node2].add(node1)

    return adj_list