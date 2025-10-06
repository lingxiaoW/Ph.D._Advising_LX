
"""
参数设置
"""

# ==== 基本控制与时间 ====
DT_GLOBAL              = 0.01    # 仿真/日志步长（秒）
MAX_TIME_PER_WP        = 1.5     # 每个随机航点的超时阈值（秒）
HOVER_AFTER_REACH      = 0.02    # 到达后悬停秒数（更紧凑可设更小或 0）
VMIN_KICK        = 0.6   # m/s，起步时的最低速度（建议 0.3~0.6，且小于 VMAX）
KICK_DIST_FACTOR = 3.0    # 仅当距离航点 > 3*WP_TOL 时启用起步加速
KP   = 3.8         # P 增益，越大“起步”越猛
VMAX = 3.4         # 最大速度上限（m/s）

# ==== 到达判定与重入 ====
WP_TOL                 = 0.10    # 到达航点的距离阈值（米）
MAX_REENTRY_PER_WP     = 2
MIN_RUN_AFTER_REENTRY  = 0.5
REENTRY_INIT_SHRINK    = 0.9
REENTRY_SHRINK_FACTOR  = 0.5

# ==== 世界范围与随机点 ====
WORLD_HALF_EXTENT      = 4.0     # 世界半边界（单位：米）
WORLD_MARGIN           = 0.30    # 距离边界的安全边距（米）
NUM_RANDOM_WP          = 10      # 随机航点个数
XY_LIM_INSET           = 1.5     # 与半边界相差的安全内缩（越大越靠近中心）
Z_MIN, Z_MAX           = 0.9, 1.4

# ==== 日志/输出 ====
CSV_FILE               = "flight_log.csv"
FIG_TRAJ_3D            = "trajectory_3d.png"
FIG_TRAJ_XY            = "trajectory_xy.png"
FIG_ALT_TIME           = "altitude_time.png"
ANIM_3D_MP4            = "trajectory_3d.mp4"
ANIM_FPS               = 30
ANIM_STRIDE            = 1       # 1=更顺滑的动画
ANIM_ELEV, ANIM_AZIM   = 25.0, -60.0
