import numpy as np
import matplotlib.pyplot as plt

# 1. 定义激活函数
def relu(x):
    """ReLU激活函数：f(x) = max(0, x)"""
    return np.maximum(0, x)

def elu(x, alpha=1.0):
    """
    ELU激活函数
    公式：当x > 0时，f(x) = x；当x <= 0时，f(x) = alpha * (e^x - 1)
    alpha：默认值为1.0（行业常用默认值）
    """
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))

# 2. 生成x轴数据（覆盖正负区间，更清晰展示函数特性）
x = np.linspace(-5, 5, 1000)  # 生成从-5到5的1000个均匀分布的点，保证曲线平滑
y_relu = relu(x)
y_elu = elu(x)

# 3. 设置绘图全局参数（重点提升清晰度）
plt.rcParams['font.sans-serif'] = ['SimHei']  # 解决中文显示问题
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题
plt.rcParams['figure.dpi'] = 150  # 提升图表分辨率（默认100，150更清晰无锯齿）
plt.rcParams['savefig.dpi'] = 300  # 若保存图片，分辨率设为300（高清）
plt.rcParams['font.size'] = 10  # 全局默认字体大小提升

# 4. 创建图表（加大尺寸，预留足够空间）
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# 5. 绘制ReLU函数（优化清晰度细节）
ax1.plot(x, y_relu, color='#2E86AB', linewidth=3, label='ReLU')  # 加粗线条
ax1.axvline(x=0, color='black', linestyle='--', alpha=0.8, linewidth=1.5)  # 加粗坐标轴虚线
ax1.axhline(y=0, color='black', linestyle='--', alpha=0.8, linewidth=1.5)
ax1.set_title('ReLU 激活函数', fontsize=16, pad=15)  # 加大标题，增加边距
ax1.set_xlabel('输入 x', fontsize=13, labelpad=10)  # 加大坐标轴标签，增加边距
ax1.set_ylabel('输出 f(x)', fontsize=13, labelpad=10)
ax1.grid(True, alpha=0.4, linewidth=0.8, linestyle='-')  # 加粗网格线，提升可见度
ax1.legend(fontsize=12, frameon=True, shadow=True)  # 加大图例，增加边框和阴影
ax1.set_xlim(-5.5, 5.5)  # 扩展坐标轴范围，避免曲线贴边
ax1.set_ylim(-1.5, 5.5)
# 标注坐标轴刻度，加大刻度字体
ax1.set_xticks(np.arange(-5, 6, 1))
ax1.set_yticks(np.arange(-1, 6, 1))
ax1.tick_params(axis='both', which='major', labelsize=11)

# 6. 绘制ELU函数（同ReLU的高清配置）
ax2.plot(x, y_elu, color='#A23B72', linewidth=3, label='ELU (α=1.0)')
ax2.axvline(x=0, color='black', linestyle='--', alpha=0.8, linewidth=1.5)
ax2.axhline(y=0, color='black', linestyle='--', alpha=0.8, linewidth=1.5)
ax2.set_title('ELU 激活函数', fontsize=16, pad=15)
ax2.set_xlabel('输入 x', fontsize=13, labelpad=10)
ax2.set_ylabel('输出 f(x)', fontsize=13, labelpad=10)
ax2.grid(True, alpha=0.4, linewidth=0.8, linestyle='-')
ax2.legend(fontsize=12, frameon=True, shadow=True)
ax2.set_xlim(-5.5, 5.5)
ax2.set_ylim(-1.5, 5.5)
ax2.set_xticks(np.arange(-5, 6, 1))
ax2.set_yticks(np.arange(-1, 6, 1))
ax2.tick_params(axis='both', which='major', labelsize=11)

# 7. 调整子图间距，显示高清图表
plt.tight_layout(pad=3.0)  # 加大子图间距，避免拥挤
plt.show()