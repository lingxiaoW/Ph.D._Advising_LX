
"""
参数文件
"""

# Geometry
Xmin, Xmax, Ymin, Ymax = -50.0, 300.0, -150.0, 150.0  # 2D 空间范围（米）
dx = dy = 2.0                                         # 网格步长（越小越细）
z_slice = 1.5                                         # 固定高度切片（米）

# Emission
SRC = (0.0, 0.0, 1.5)  # 源位置 (x, y, z)
Q_PUFF = 50.0          # 单个 puff 的相对量

# 风场
U_MEAN = 2.0           # 平均风速 (m/s)
DIR0_DEG = 0.0         # 基准风向（度）：0°=沿 +x
RW_SIGMA_DEG_PER_S = 1.2   # 每秒风向增量的标准差（度/秒）
RW_MAX_AMP_DEG = 45.0      # 最大偏离角（±度）

# 扩散系数（sigma ~ a*sqrt(t)）
A_Y = 1.2  # 横向（影响 y 扩展）
A_Z = 0.8  # 竖向（含地面镜像影响）

#Time params
SIM_DT  = 1.0          # 步长（秒）
PUFF_DT = 5.0          # puff 释放间隔（秒）
TOTAL_T = 6 * 60.0     # 总时长（秒）
OUT_EVERY = 5          # 输出间隔 = OUT_EVERY * SIM_DT

#可视化
GIF_NAME = "puff_wind_randomwalk_test.gif"
PERC_FOR_VMAX = 99.5   # 每帧色标上限取该分位数，增强对比度
