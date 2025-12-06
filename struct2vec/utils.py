# -*- coding: utf-8 -*-
from time import time
import logging, inspect
import pickle as pickle
from itertools import islice
import os

# 获取当前脚本所在目录
dir_f = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
# 构建 pickles 文件夹的绝对路径 (兼容 Windows/Linux)
folder_pickles = os.path.abspath(os.path.join(dir_f, "..", "pickles"))


def returnPathStruc2vec():
    return dir_f


def isPickle(fname):
    # 使用 os.path.join 拼接路径
    path = os.path.join(folder_pickles, fname + '.pickle')
    return os.path.isfile(path)


def chunks(data, SIZE=10000):
    it = iter(data)
    for i in range(0, len(data), SIZE):
        yield {k: data[k] for k in islice(it, SIZE)}


def partition(lst, n):
    division = len(lst) / float(n)
    return [lst[int(round(division * i)): int(round(division * (i + 1)))] for i in range(n)]


def restoreVariableFromDisk(name):
    logging.info('Recovering variable...')
    t0 = time()
    val = None

    # 路径拼接
    path = os.path.join(folder_pickles, name + '.pickle')

    # 增加文件存在性检查，防止误报
    if not os.path.exists(path):
        # 尝试创建一个空的pickles文件夹防止后续报错，或者抛出更清晰的错误
        if not os.path.exists(folder_pickles):
            os.makedirs(folder_pickles)
        logging.error(f"File not found: {path}")
        return None

    with open(path, 'rb') as handle:
        val = pickle.load(handle)
    t1 = time()
    logging.info('Variable recovered. Time: {}m'.format((t1 - t0) / 60))

    return val


def saveVariableOnDisk(f, name):
    logging.info('Saving variable on disk...')
    t0 = time()

    # --- 如果没有 pickles 文件夹，自动创建 ---
    if not os.path.exists(folder_pickles):
        try:
            os.makedirs(folder_pickles)
            logging.info(f"Created directory: {folder_pickles}")
        except OSError as e:
            logging.error(f"Failed to create directory {folder_pickles}: {e}")
    # ------------------------------------------------

    path = os.path.join(folder_pickles, name + '.pickle')

    with open(path, 'wb') as handle:
        pickle.dump(f, handle, protocol=pickle.HIGHEST_PROTOCOL)
    t1 = time()
    logging.info('Variable saved. Time: {}m'.format((t1 - t0) / 60))

    return