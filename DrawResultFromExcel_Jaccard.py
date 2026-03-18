import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_topk_overlap_from_excel(excel_path):
    """
    读取 Excel 中的 Jaccard 列，按网络类型聚合，绘制 Top-K Overlap 折线图
    证明 ListMLE/Listwise 排序在头部预测的优势。
    """
    print(f"正在读取数据: {excel_path}")
    df = pd.read_excel(excel_path)

    # 1. 提取网络的主类型 (例如：从 'BA_500_0' 提取出 'BA')
    df['Net_Type'] = df['Network'].astype(str).apply(lambda x: x.split('_')[0])

    # 我们要处理的 Top-K 比例列
    x_labels = ['10%', '20%', '30%', '40%', '50%']
    jaccard_cols = [f'Jaccard_{p}' for p in x_labels]

    # 检查列是否存在
    missing_cols = [col for col in jaccard_cols if col not in df.columns]
    if missing_cols:
        print(f"错误: Excel 中缺失以下列: {missing_cols}")
        return

    # 2. 聚合数据：按 'Net_Type' 和 'Method' 对所有规模的 Jaccard 值求平均
    agg_df = df.groupby(['Net_Type', 'Method'])[jaccard_cols].mean().reset_index()
    net_types = sorted(agg_df['Net_Type'].unique())

    if not net_types:
        print("未找到有效的网络类型数据，请检查 Excel 内容。")
        return

    # 3. 动态布局计算 (每个网络类型画一个子图)
    n_plots = len(net_types)
    n_cols = 3
    n_rows = (n_plots + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 6 * n_rows))
    if n_rows * n_cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    # 4. 样式字典 (复用之前的样式体系)
    methods = [
        "GDN_Model", "Degree", "H-index", "Betweenness", "Eigenvector",
        "RCNN", "InfGCN", "CGNN", "CNT", "AGNN"
    ]
    m_colors = {
        "GDN_Model": "tab:red", "Degree": "tab:gray", "H-index": "tab:blue",
        "Betweenness": "tab:green", "Eigenvector": "tab:orange",
        "RCNN": "tab:purple", "InfGCN": "tab:brown", "CGNN": "tab:pink",
        "CNT": "tab:olive", "AGNN": "tab:cyan"
    }
    m_styles = {
        "GDN_Model": "-", "Degree": ":", "H-index": ":", "Betweenness": "-.", "Eigenvector": "-.",
        "RCNN": "--", "InfGCN": "--", "CGNN": "--", "CNT": (0, (5, 2)), "AGNN": (0, (5, 2))
    }
    m_markers = {
        "GDN_Model": "*", "Degree": "o", "H-index": "v", "Betweenness": "^", "Eigenvector": "<",
        "RCNN": "s", "InfGCN": "D", "CGNN": "P", "CNT": "X", "AGNN": "h"
    }

    # 5. 循环绘制每个子图
    x_positions = [10, 20, 30, 40, 50]  # 用作绘图的 X 坐标

    for i, t in enumerate(net_types):
        ax = axes[i]
        # 筛选当前网络类型的数据
        sub_df = agg_df[agg_df['Net_Type'] == t]

        methods_plotted = False
        for m_name in methods:
            # 获取该方法在该网络下的数据
            m_data = sub_df[sub_df['Method'] == m_name]
            if m_data.empty:
                continue

            # 提取 5 个百分比的平均值
            y_values = m_data[jaccard_cols].values[0].tolist()

            # 如果存在全 NaN 的情况则跳过
            if all(pd.isna(v) for v in y_values):
                continue

            is_model = (m_name == "GDN_Model")
            lw = 2.5 if is_model else 1.5
            ms = 10 if is_model else 6
            alpha = 1.0 if is_model else 0.8
            zorder = 10 if is_model else 1  # 保证 GDN_Model 画在最上层，不被遮挡

            ax.plot(x_positions, y_values,
                    label=m_name,
                    color=m_colors.get(m_name, "black"),
                    linestyle=m_styles.get(m_name, "-"),
                    marker=m_markers.get(m_name, "o"),
                    linewidth=lw,
                    markersize=ms,
                    alpha=alpha,
                    zorder=zorder)
            methods_plotted = True

        ax.set_title(f"Top-K Overlap Analysis: {t} Networks", fontweight='bold', fontsize=14)
        ax.set_xlabel("Top K (%)", fontsize=12)
        ax.set_ylabel("Overlap@K / Precision", fontsize=12)

        # 设定 X 轴刻度为 10, 20, 30, 40, 50
        ax.set_xticks(x_positions)
        ax.set_xticklabels(x_labels)

        ax.set_ylim(0, 1.0)  # Jaccard/Overlap 范围通常是 0 到 1
        ax.grid(True, linestyle='--', alpha=0.5)

        if methods_plotted:
            # 图例放在左下角或右下角均可，根据曲线走势，通常重合度随 K 增大而趋稳
            ax.legend(ncol=2, fontsize=8, loc='best')

    # 隐藏多余子图
    for j in range(len(net_types), len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # 请确保此时 Excel 里已经包含了那 5 列 Jaccard 数据
    excel_file = "results/evaluation_results_epoch1287_re.xlsx"
    plot_topk_overlap_from_excel(excel_file)