import matplotlib.pyplot as plt
import numpy as np

# 设置字体：中文宋体，英文新罗马
plt.rcParams['font.sans-serif'] = ['SimSun']  # 宋体用于中文
plt.rcParams['font.serif'] = ['Times New Roman']  # 新罗马用于英文
plt.rcParams['axes.unicode_minus'] = False    # 用于显示负号
plt.rcParams['font.size'] = 16                # 全局字体大小放大2号（原14→16）

# 数据
methods = ['BadNets', 'Blend', 'WaNet', 'FEAT', 'Ours']
anomaly_indices = [4.6, 5.5, 4.1, 3.9, 1.9]
threshold = 4.0  # 阈值线

# 创建图形（画布宽度从10收窄至7.5）
fig, ax = plt.subplots(figsize=(7.5, 6))

# 绘制柱状图（宽度为原来的2/3，默认宽度约为0.8，改为0.53）
# 马卡龙配色，饱和度较低
bars = ax.bar(methods, anomaly_indices, width=0.53,
              color=['#A8D5E5', '#F4B9B2', '#C5E0D4', '#FFD9B5', '#E0C3E5'],
              edgecolor='#888888', linewidth=1.0)

# 添加阈值线
ax.axhline(y=threshold, color='#E67E22', linestyle='--', linewidth=2, label=f'阈值 ({threshold})')

# 设置标签和标题（字号放大2号：原18→20）
ax.set_ylabel('异常指数', fontsize=20, fontweight='bold')
ax.set_xlabel('攻击方法', fontsize=20, fontweight='bold')

# 设置y轴范围
ax.set_ylim(0, max(anomaly_indices) + 1.0)

# 添加网格线（y轴方向，虚线）
ax.grid(axis='y', linestyle='--', alpha=0.6)

# 添加图例，字号放大（原16→18）
ax.legend(loc='upper right', fontsize=18)

# 设置坐标轴刻度字体大小（原16→18）
ax.tick_params(axis='both', labelsize=18)

# 调整布局，确保标签完整显示
plt.tight_layout()

# 保存图像
plt.savefig('anomaly_index_comparison.png', dpi=300, bbox_inches='tight')
print("图像已保存为 'anomaly_index_comparison.png'")

# 显示图像（如果需要）
# plt.show()