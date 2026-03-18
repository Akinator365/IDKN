import collections
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter


def plot_results_from_excel(excel_path):
    """
    从导出的 Excel 文件读取结果并绘制：
    1. 左图: BA 网络参数分析 (线性轴)
    2. 右图: 全网络规模对比 (对数横坐标轴)
    3. 独立对比图 (Model vs Baselines)
    """
    print(f"正在读取数据: {excel_path}")
    df = pd.read_excel(excel_path)

    # 初始化数据容器 (与原逻辑保持一致)
    ba_param_data = collections.defaultdict(lambda: collections.defaultdict(list))
    comparison_data = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(list)))

    # --- 1. 数据解析 ---
    # 遍历 DataFrame 的每一行
    for _, row in df.iterrows():
        name = row['Network']
        m_name = row['Method']
        tau = row['Kendall_Tau']

        # 排除空值
        if pd.isna(tau):
            continue

        parts = str(name).split('_')
        if len(parts) < 2:
            continue

        net_type = parts[0]
        try:
            size = int(parts[1])
        except ValueError:
            continue

        # 存储所有类型、所有方法的规模对比数据
        comparison_data[net_type][m_name][size].append(tau)

        # 专门存储 BA 的参数敏感性数据 (仅限模型)
        if net_type == 'BA' and m_name == "GDN_Model" and len(parts) >= 3:
            try:
                # 按照你原代码的逻辑，取 parts[2] 作为 m 值
                m_val = int(parts[2])
                ba_param_data[size][m_val].append(tau)
            except ValueError:
                pass

    # --- 2. 动态布局计算 ---
    net_types = sorted(comparison_data.keys())
    if not net_types:
        print("未在 Excel 中找到有效数据，请检查文件内容。")
        return

    n_plots = 2 + len(net_types)
    n_cols = 3
    n_rows = (n_plots + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8 * n_cols, 7 * n_rows))
    # 兼容只有一行子图的情况
    if n_rows * n_cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    # 公用格式化器
    def set_log_xaxis(ax, sizes):
        if not sizes: return
        ax.set_xscale('log')
        fmt = ScalarFormatter()
        fmt.set_scientific(False)
        ax.xaxis.set_major_formatter(fmt)
        ax.set_xticks(sorted(list(sizes)))
        ax.grid(True, which='both', linestyle='--', alpha=0.4)

    # === 图 1: BA 网络参数分析 (保持线性) ===
    ax1 = axes[0]
    ba_sizes = sorted(ba_param_data.keys())
    ba_params = [3, 5, 8, 15]  # 你原代码中固定的参数
    colors_ba = plt.cm.viridis(np.linspace(0, 0.8, max(1, len(ba_sizes))))

    for i, size in enumerate(ba_sizes):
        x, y = [], []
        for m in ba_params:
            vals = ba_param_data[size].get(m, [])
            if vals:
                x.append(m)
                y.append(np.nanmean(vals))
        if x:
            ax1.plot(x, y, marker='o', label=f"Size {size}", linewidth=2, color=colors_ba[i])

    ax1.set_title("Analysis 1: BA Parameter Sensitivity", fontsize=14, fontweight='bold')
    ax1.set_xlabel("m", fontsize=12)
    ax1.set_ylabel("Tau", fontsize=12)
    ax1.set_ylim(0.2, 1.0)
    if ba_sizes: ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)

    # === 图 2: 全网络规模对比 (原图2 - 模型合集) ===
    ax2 = axes[1]
    markers = ['o', 's', '^', 'D', 'v', 'X', 'P', '*']
    all_sizes = set()

    for idx, t in enumerate(net_types):
        size_dict = comparison_data[t].get("GDN_Model", {})
        sorted_sizes = sorted(size_dict.keys())
        x, y = [], []
        for s in sorted_sizes:
            x.append(s)
            y.append(np.nanmean(size_dict[s]))
            all_sizes.add(s)
        if x:
            ax2.plot(x, y, marker=markers[idx % len(markers)], label=t, linewidth=2)

    set_log_xaxis(ax2, all_sizes)
    ax2.set_title("Analysis 2: All Networks Scale (GDN Only)", fontweight='bold')
    ax2.set_xlabel("Size (N)")
    ax2.set_ylabel("Tau")
    ax2.set_ylim(0.2, 1.0)
    if all_sizes: ax2.legend()

    # === 图 3 及以后: 独立对比图 (Model vs Baselines) ===
    methods = [
        "GDN_Model", "Degree", "H-index", "Betweenness", "Eigenvector",
        "RCNN", "InfGCN", "CGNN", "CNT", "AGNN"
    ]
    m_colors = {
        # 原有的经典指标和你的模型
        "GDN_Model": "tab:red",
        "Degree": "tab:gray",
        "H-index": "tab:blue",
        "Betweenness": "tab:green",
        "Eigenvector": "tab:orange",
        # 新增的深度学习/其他基线方法
        "RCNN": "tab:purple",
        "InfGCN": "tab:brown",
        "CGNN": "tab:pink",
        "CNT": "tab:olive",
        "AGNN": "tab:cyan"
    }

    # 2. 线型字典 (区分三大阵营)
    m_styles = {
        "GDN_Model": "-",  # 你的模型：实线 (最醒目)

        "Degree": ":",  # 传统方法：点线
        "H-index": ":",
        "Betweenness": "-.",  # 传统方法：点划线
        "Eigenvector": "-.",

        "RCNN": "--",  # 深度学习方法：标准虚线
        "InfGCN": "--",
        "CGNN": "--",
        "CNT": (0, (5, 2)),  # 深度学习方法：长虚线 (自定义，5实2空)
        "AGNN": (0, (5, 2))
    }

    # 3. 标记点字典 (进一步防伪，不同方法不同点)
    m_markers = {
        "GDN_Model": "*",  # 你的模型：大五角星

        "Degree": "o",  # 传统方法：圆形、下三角、上三角、左三角
        "H-index": "v",
        "Betweenness": "^",
        "Eigenvector": "<",

        "RCNN": "s",  # 深度学习方法：正方形、菱形、加号、乘号、六边形
        "InfGCN": "D",
        "CGNN": "P",
        "CNT": "X",
        "AGNN": "h"
    }

    for i, t in enumerate(net_types):
        ax = axes[i + 2]
        sizes_seen = set()

        for m_name in methods:
            if m_name not in comparison_data[t]: continue
            size_dict = comparison_data[t][m_name]
            sorted_s = sorted(size_dict.keys())
            x, y = [], []
            for s in sorted_s:
                x.append(s)
                y.append(np.nanmean(size_dict[s]))
                sizes_seen.add(s)

            if x:
                # 动态调整线宽和透明度：突出主角 GDN_Model
                is_model = (m_name == "GDN_Model")
                lw = 2.5 if is_model else 1.5
                ms = 10 if is_model else 6
                alpha = 1.0 if is_model else 0.8  # 基线稍微变淡一点点，不抢主线风头

                ax.plot(x, y,
                        label=m_name,
                        color=m_colors.get(m_name, "black"),
                        linestyle=m_styles.get(m_name, "-"),
                        marker=m_markers.get(m_name, "o"),
                        linewidth=lw,
                        markersize=ms,
                        alpha=alpha)

        set_log_xaxis(ax, sizes_seen)
        ax.set_title(f"Analysis {i + 3}: {t} Comparison", fontweight='bold')
        ax.set_xlabel("Size (N)")
        ax.set_ylabel("Tau")
        ax.set_ylim(0.2, 1.0)

        # 调整图例：如果有 10 个方法，可以把图例分成两列显示，避免挡住图表
        if sizes_seen:
            ax.legend(ncol=2, fontsize=8, loc='lower right')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # 调用方式：传入刚才生成的 Excel 文件即可
    excel_file = "results/evaluation_results_epoch1287_re.xlsx"
    plot_results_from_excel(excel_file)