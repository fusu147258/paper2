import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")  # 在导入 pyplot 前设置 backend
import matplotlib.pyplot as plt
# 设置分辨率（8x8 是典型块大小）
N = 8
k1 = np.arange(N)
k2 = np.arange(N)

# 创建频率网格
K1, K2 = np.meshgrid(k2, k1)  # 注意：这里我们把 k1 放在 y 轴，k2 放在 x 轴

# 计算频率能量（基于 k1 和 k2 的平方和）
frequency_map = np.sqrt(K1**2 + K2**2)

# 归一化到 [0, 1] 并映射为颜色
normalized_freq = frequency_map / np.max(frequency_map)

# 使用 colormap 映射：红色=低频，蓝色=高频
cmap = plt.cm.coolwarm  # 或用 'hot'、'plasma'、'RdYlBu'
color_map = cmap(normalized_freq)

# 可选：将 RGB 通道分离并转换为整数
color_map_uint8 = (color_map[:, :, :3] * 255).astype(np.uint8)

# 保存图像
plt.figure(figsize=(4, 4))
plt.imshow(color_map_uint8, origin='lower', cmap='coolwarm')
plt.colorbar(label="Frequency", shrink=0.8)
plt.xlabel(r'$k_2$')
plt.ylabel(r'$k_1$')
plt.title('Frequency Map of DCT')
plt.tight_layout()
plt.savefig('dct_frequency_map.png', dpi=300, bbox_inches='tight')
plt.show()

# 也可以直接保存为 PNG（无坐标轴）
plt.figure(figsize=(4, 4))
plt.imshow(color_map_uint8, origin='lower', cmap='coolwarm')
plt.axis('off')
plt.savefig('dct_frequency_map_clean.png', dpi=300, bbox_inches='tight', pad_inches=0)
plt.close()