import pickle
import pandas as pd


def pkl_to_excel(pkl_file_path, output_excel_path):
    print(f"正在读取文件: {pkl_file_path}")

    # 1. 加载 pkl 文件
    with open(pkl_file_path, 'rb') as f:
        results = pickle.load(f)

    data_rows = []

    # 2. 解析嵌套字典结构
    # 结构: results[network_name][method_name] = {"Tau": [val], "MI": [val], "Jaccard": [[v1, v2, v3, v4, v5]]}
    for network_name, methods in results.items():
        for method_name, metrics in methods.items():

            # 取列表中的最后一个值（[-1]），兼容单个网络被多次 evaluate 追加的情况
            tau = metrics.get("Tau", [None])[-1]
            mi = metrics.get("MI", [None])[-1]

            # Jaccard 提取嵌套列表的最后一项
            jaccard_list = metrics.get("Jaccard", [[None] * 5])[-1]

            # 构建单行数据
            row = {
                "Network": network_name,
                "Method": method_name,
                "Kendall_Tau": tau,
                "MI_Monotonicity": mi
            }

            # 根据你 compute_metrics 中的 percentages = [0.1, 0.2, 0.3, 0.4, 0.5] 展开 Jaccard
            percentages = [10, 20, 30, 40, 50]
            for i, p in enumerate(percentages):
                # 防止由于某些原因 Jaccard 列表长度不够导致报错
                val = jaccard_list[i] if i < len(jaccard_list) else None
                row[f"Jaccard_{p}%"] = val

            data_rows.append(row)

    # 3. 转换为 DataFrame
    df = pd.DataFrame(data_rows)

    # 可选：按照 Network 和 Kendall_Tau 进行排序，方便在 Excel 中直接对比哪个方法更好
    df = df.sort_values(by=["Network", "Kendall_Tau"], ascending=[True, False])

    # 4. 导出为 Excel
    df.to_excel(output_excel_path, index=False)
    print(f"数据已成功转换并保存至: {output_excel_path}")


if __name__ == '__main__':
    # 替换为你实际的 pkl 文件名
    input_pkl = "result_2025-12-31_14-37-07_epoch1287.pkl"
    output_xlsx = "evaluation_results_epoch1287.xlsx"

    pkl_to_excel(input_pkl, output_xlsx)