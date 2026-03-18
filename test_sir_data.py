import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import warnings

warnings.filterwarnings('ignore')


# -------------------------- 1. 数据读取函数 --------------------------
def load_sir_data(file_path):
    """
    读取SIR模拟的节点影响力数据
    """
    try:
        df = pd.read_csv(
            file_path,
            sep='\t',
            header=None,
            names=['node_id', 'influence'],
            dtype={'node_id': int, 'influence': float}
        )

        # 【修正】: 移除 3σ 剔除逻辑，BA网络保留长尾高影响力节点
        # 仅做基础清洗：去重、去负值
        df = df.drop_duplicates(subset=['node_id'])
        df = df[df['influence'] >= 0]

        df = df.sort_values('node_id').reset_index(drop=True)
        print(f"数据读取完成！共{len(df)}个节点")
        return df
    except Exception as e:
        print(f"数据读取失败：{e}")
        return None


# -------------------------- 2. BA网络SIR评估核心函数 --------------------------
def evaluate_ba_sir(df):
    """
    基于BA网络SIR特性的多维度评估
    """
    influence = df['influence'].values
    results = {}

    # -------- 维度1：基础统计（新增：方差、最大最小） --------
    results['mean_influence'] = np.mean(influence)
    results['median_influence'] = np.median(influence)
    results['max_value'] = np.max(influence)  # 新增
    results['min_value'] = np.min(influence)  # 新增
    results['variance'] = np.var(influence)  # 新增
    results['std_dev'] = np.std(influence)  # 标准差

    # 防止分母为0
    min_val = results['min_value']
    results['max_min_ratio'] = results['max_value'] / min_val if min_val > 0 else 0

    # -------- 维度2：异质性（核心指标） --------
    # 变异系数 CV (标准差/均值)
    mean_val = results['mean_influence']
    results['coefficient_variation'] = results['std_dev'] / mean_val if mean_val > 0 else 0

    # 四分位距系数
    q75, q25 = np.percentile(influence, [75, 25])
    median_val = results['median_influence']
    results['iqr_coefficient'] = (q75 - q25) / median_val if median_val > 0 else 0

    # -------- 维度3：稳定性 --------
    # 波动系数
    results['fluctuation_coefficient'] = results['coefficient_variation'] / np.sqrt(len(influence))

    # -------- 维度4：头部聚集效应 (帕累托原则) --------
    # 计算前10%的节点占据了总影响力的多少
    top10_percent_idx = int(len(influence) * 0.1)
    if top10_percent_idx < 1: top10_percent_idx = 1

    # 降序排列
    sorted_inf_desc = np.sort(influence)[::-1]
    top10_sum = np.sum(sorted_inf_desc[:top10_percent_idx])
    total_sum = np.sum(influence)

    results['top10_ratio'] = top10_sum / total_sum if total_sum > 0 else 0
    results['top10_count_ratio'] = 0.1  # 固定记录一下

    # -------- 维度5：幂律分布拟合 --------
    try:
        # 使用 CCDF (互补累积分布函数) 进行拟合
        sorted_inf = np.sort(influence)
        y_vals = 1.0 - np.arange(len(sorted_inf)) / len(sorted_inf)

        # 剔除0值和尾部噪音
        mask = (sorted_inf > 0) & (y_vals > 0)
        x_log = np.log10(sorted_inf[mask])
        y_log = np.log10(y_vals[mask])

        if len(x_log) >= 10:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x_log, y_log)
            results['power_law_slope'] = slope
            results['power_law_r2'] = r_value ** 2
        else:
            results['power_law_slope'] = 0
            results['power_law_r2'] = 0
    except Exception as e:
        results['power_law_slope'] = 0
        results['power_law_r2'] = 0

    return results


# -------------------------- 3. 结果可视化函数 --------------------------
def visualize_results(df, eval_results):
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    influence = df['influence'].values

    # 子图1：直方图
    ax1 = axes[0, 0]
    ax1.hist(influence, bins=40, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.axvline(np.mean(influence), color='red', linestyle='--', label='Mean')
    ax1.axvline(np.median(influence), color='orange', linestyle='--', label='Median')
    ax1.set_title('Influence Distribution')
    ax1.legend()

    # 子图2：双对数CCDF
    ax2 = axes[0, 1]
    sorted_inf = np.sort(influence)
    y_vals = 1.0 - np.arange(len(sorted_inf)) / len(sorted_inf)
    mask = (sorted_inf > 0)

    x_log = np.log10(sorted_inf[mask])
    y_log = np.log10(y_vals[mask])

    ax2.scatter(x_log, y_log, s=10, color='darkblue', alpha=0.6, label='Data')

    if eval_results.get('power_law_r2', 0) > 0:
        slope = eval_results['power_law_slope']
        intercept = np.mean(y_log) - slope * np.mean(x_log)
        fit_y = slope * x_log + intercept
        ax2.plot(x_log, fit_y, color='red', linewidth=2,
                 label=f'Fit (Slope={slope:.2f}, R2={eval_results["power_law_r2"]:.3f})')

    ax2.set_xlabel('log10(Influence)')
    ax2.set_ylabel('log10(CCDF)')
    ax2.set_title('Power-law Check (CCDF)')
    ax2.legend()
    ax2.grid(alpha=0.3)

    # 子图3：箱线图
    ax3 = axes[1, 0]
    box_plot = ax3.boxplot(influence, patch_artist=True)
    ax3.set_title('Box Plot (Heterogeneity)')
    ax3.grid(alpha=0.3, axis='y')

    # 子图4：文本汇总（这里也更新一下显示）
    ax4 = axes[1, 1]
    ax4.axis('off')
    text_str = f"""关键指标概览:

    [基础统计]
    Mean: {eval_results['mean_influence']:.4f}
    Max:  {eval_results['max_value']:.4f}
    Min:  {eval_results['min_value']:.4f}
    Var:  {eval_results['variance']:.6f}

    [BA特性验证]
    CV (异质性): {eval_results['coefficient_variation']:.4f}
    Top10% 占比: {eval_results['top10_ratio']:.4f}
    R2 (幂律拟合): {eval_results.get('power_law_r2', 0):.3f}
    """
    ax4.text(0.05, 0.95, text_str, transform=ax4.transAxes, fontsize=12,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout()
    plt.show()


# -------------------------- 4. 详细结果打印函数 --------------------------
def print_results(eval_results):
    """
    打印所有计算的指标
    """
    print("\n" + "=" * 60)
    print("SIR模拟结果详细评估报告")
    print("=" * 60)

    print(f"\n【1. 基础统计指标】")
    print(f"  • 最大值 (Max):      {eval_results['max_value']:.8f}")
    print(f"  • 最小值 (Min):      {eval_results['min_value']:.8f}")
    print(f"  • 极值比 (Max/Min):  {eval_results['max_min_ratio']:.2f}")
    print(f"  • 平均值 (Mean):     {eval_results['mean_influence']:.8f}")
    print(f"  • 中位数 (Median):   {eval_results['median_influence']:.8f}")
    print(f"  • 方差 (Variance):   {eval_results['variance']:.8f}")
    print(f"  • 标准差 (Std Dev):  {eval_results['std_dev']:.8f}")

    print(f"\n【2. 异质性与不平等度 (BA网络核心特性)】")
    print(f"  • 离散系数 (CV):     {eval_results['coefficient_variation']:.4f} (CV > 1 通常表示强异质性)")
    print(f"  • 四分位距系数:      {eval_results['iqr_coefficient']:.4f}")
    print(f"  • 头部聚集度:        前10%的节点贡献了 {eval_results['top10_ratio'] * 100:.2f}% 的总影响力")

    print(f"\n【3. 分布特性检验】")
    r2 = eval_results.get('power_law_r2', 0)
    slope = eval_results.get('power_law_slope', 0)
    print(f"  • 幂律拟合优度 R²:   {r2:.4f}")
    print(f"  • 拟合斜率 (CCDF):   {slope:.4f} (理论上应接近 -2，对应度分布指数 -3)")

    print(f"\n【4. 稳定性指标】")
    print(f"  • 波动系数:          {eval_results['fluctuation_coefficient']:.8f}")

    print("=" * 60)

# -------------------------- 4. 结果解读函数 (逻辑修正) --------------------------
def interpret_results(eval_results):
    """
    动态解读结果，不使用硬性阈值
    """
    print("\n" + "=" * 60)
    print("评估结果解读（基于BA网络SIR特性）：")
    print("=" * 60)

    # 1. 异质性判断
    cv = eval_results['coefficient_variation']
    top10 = eval_results['top10_ratio']

    print(f"\n【1. 网络异质性】")
    if cv > 1.0 or top10 > 0.4:
        print(f"  ✅ 数据显示出强烈的异质性 (CV={cv:.2f})，前10%节点贡献了 {top10 * 100:.1f}% 的影响力。")
        print("     这符合BA网络中Hub节点主导传播的特征。")
    else:
        print(f"  ⚠️ 数据异质性较弱 (CV={cv:.2f})，影响力分布较为均匀。")
        print("     需检查SIR参数(beta)是否过大导致全体爆发，掩盖了网络结构差异。")

    # 2. 幂律特性
    r2 = eval_results.get('power_law_r2', 0)
    slope = eval_results.get('power_law_slope', 0)

    print(f"\n【2. 无标度特性验证】")
    if r2 > 0.8:
        print(f"  ✅ 影响力分布在双对数坐标下呈现良好的线性 (R2={r2:.3f})。")
        print(f"     拟合斜率为 {slope:.2f}。")
    else:
        print(f"  ❌ 影响力分布未呈现明显的幂律特征 (R2={r2:.3f})。可能原因：数据量不足或传播已饱和。")


# 调用示例（替换为你的数据文件路径）
if __name__ == "__main__":

    TRAIN_LABELS_PATH = os.path.join(os.getcwd(), 'data', 'labels', 'train')
    REALWORLD_LABELS_PATH = os.path.join(os.getcwd(), 'data', 'labels', 'realworld')
    # network_type = "LFR"
    # network = "LFR_500"
    # network_no = 1
    # network_name = network + "_" + str(network_no)
    # labels_path = os.path.join(TRAIN_LABELS_PATH, network_type + '_graph', network, network_name + '_labels.txt')
    # realworld_name = "AirTraffic"
    # labels_path = os.path.join(REALWORLD_LABELS_PATH, realworld_name + '_labels.txt')

    # 2. 读取数据
    df = load_sir_data(labels_path)
    if df is None:
        exit()

    # 3. 执行评估
    eval_results = evaluate_ba_sir(df)

    # 4. 可视化结果
    visualize_results(df, eval_results)

    print_results(eval_results)

    # 5. 解读结果
    interpret_results(eval_results)