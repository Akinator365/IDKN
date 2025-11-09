from scipy import stats
import sys
import os

# --- 1. 在这里修改参数 ---
RUN1_ID = 1  # (要比较的第一个 run_id)
RUN2_ID = 2  # (要比较的第二个 run_id)
# -------------------------

# --- 2. 固定路径设置 ---
BASE_DIR = os.getcwd()
# NETWORK_NAME = "karate_club_graph"
NETWORK_NAME = "Jazz"
RESULTS_BASE_PATH = os.path.join(BASE_DIR, 'data', 'labels', 'realworld')


def read_results_file(filepath):
    """
    读取模拟结果文件。
    返回一个字典 {node_id: score}
    """
    scores = {}
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('#') or line.strip() == '':
                continue
            try:
                parts = line.split()
                node_id = int(parts[0])
                score = float(parts[1])
                scores[node_id] = score
            except (IndexError, ValueError):
                print(f"警告：跳过格式错误的行: {line.strip()}")

    if not scores:
        print(f"错误：文件 {filepath} 为空或无法解析。")
        return None

    return scores



def calculate_kendall(file1_path, file2_path):
    print(f"--- 正在比较文件 ---")
    print(f"文件 1: {file1_path}")
    print(f"文件 2: {file2_path}")

    scores1 = read_results_file(file1_path)
    scores2 = read_results_file(file2_path)

    if scores1 is None or scores2 is None:
        print("计算中止。")
        return

    nodes1 = set(scores1.keys())
    nodes2 = set(scores2.keys())

    if nodes1 != nodes2:
        print("错误：两个文件的节点集不匹配。")
        return

    sorted_nodes = sorted(list(nodes1))

    rank_list1 = [scores1[node] for node in sorted_nodes]
    rank_list2 = [scores2[node] for node in sorted_nodes]

    print(f"成功加载 {len(sorted_nodes)} 个节点的分数。")

    tau, p_value = stats.kendalltau(rank_list1, rank_list2)

    print("\n--- 计算结果 ---")
    print(f"肯德尔相关系数 (Kendall's Tau): {tau:.6f}")
    print(f"P 值 (P-value): {p_value:.6f}")

    if p_value < 0.05:
        print(f"结论：两个排名之间存在统计上显著的相关性 (Tau={tau:.4f})。")
    else:
        print("结论：两个排名之间的相关性在统计上不显著。")


# --- 3. 运行 ---
if __name__ == "__main__":
    # 根据硬编码的参数构造文件路径
    file1_name = f"{NETWORK_NAME}_labels_{RUN1_ID}.txt"
    file2_name = f"{NETWORK_NAME}_labels_{RUN2_ID}.txt"

    file1_path = os.path.join(RESULTS_BASE_PATH, file1_name)
    file2_path = os.path.join(RESULTS_BASE_PATH, file2_name)

    # 运行计算
    calculate_kendall(file1_path, file2_path)