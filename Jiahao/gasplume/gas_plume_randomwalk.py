"""
根据FastGaussianPuff修改的简化版：
https://github.com/Hammerling-Research-Group/FastGaussianPuff
用 Python 实现“Fast Implementation of the Gaussian Puff Forward Atmospheric Model”

【与 FGP 的概念映射（Concept mapping to FGP）】
- Geometry（几何）：
  - FGP：支持规则网格与“稀疏点（points-in-space）”两类几何。
  - 本脚本：使用规则 2D 网格的“固定高度切片 z = const”来绘制热图（可看作 grid-geometry 的一个切片）。
- Emission（排放/源）：
  - FGP：Emission 对象包含源位置、排放率、起止时刻等；以固定间隔生成 puff（puff_dt）。
  - 本脚本：SRC 为源位置；Q_PUFF 为单个 puff 的“量”；PUFF_DT 控制释放间隔（t_since_emit 达阈值则释放）。
- Wind series（风场/时间序列）：
  - FGP：接受时间戳等间隔的风速/风向序列，内部插值到仿真步长。
  - 本脚本：通过 wind_vec_at(t) 构造“随时间变化的风向”，含两种模式（正弦摆动/随机游走）；可轻松改成读取你的 CSV。
- Time parameters（时间参数）：
  - FGP：sim_dt（仿真步）、puff_dt（脉冲发射间隔）、obs_dt/output_dt（观测/输出间隔）。
  - 本脚本：SIM_DT、PUFF_DT、OUT_EVERY（每几步输出一帧）对应 sim_dt/puff_dt/output_dt 的思想。
- Physics/Kernel（核心核函数）：
  - FGP：C++ 高性能核；含阈值裁剪、低风跳过（skip_low_wind）、exp_thresh_tolerance、unsafe 等加速/近似开关。
  - 本脚本：纯 Python 的 cp_concentration() 叠加，不做阈值裁剪与近似开关（方便阅读与验证逻辑）。
- I/O/Plot（输出/可视化）：
  - FGP：demo 里用 Matplotlib/GIF；常见的图形后端在 Linux/macOS 友好，Windows 建议离屏渲染。
  - 本脚本：强制使用 Agg 离屏后端（matplotlib.use("Agg")），避免 Windows/WSL 的 Qt/Wayland 问题。

【可迁移性/升级建议】
- 想接 FGP：保留“概念一致”的函数签名与数据组织（Geometry/Emission/Wind/TimeParams），把 cp_concentration 与 advect_and_age
  等替换为 FGP 暴露的 Python 绑定调用（C++ 后端）。
- 想提速（Windows 仍纯 Python）：给 cp_concentration/叠加循环加 numba.njit 并做批量向量化。
"""

import os
import math
import numpy as np
import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")  # 【与 FGP 的可视化实践一致】离屏渲染，避免 GUI 依赖/后端冲突（Windows/WSL 更稳）
import matplotlib.pyplot as plt

# 1)可调参数

# --- Geometry: 规则网格 + 固定高度切片（对应 FGP 的 grid 几何，常用于体场/切片可视化）
Xmin, Xmax, Ymin, Ymax = -50.0, 300.0, -150.0, 150.0   # 2D 空间范围（m）
dx = dy = 2.0                                          # 网格步长（越小越细致）
z_slice = 1.5                                          # 固定可视化高度（m）

# --- Emission: 定时脉冲（对应 FGP.Emission）
SRC = np.array([0.0, 0.0, 1.5])  # 源位置 (x, y, z)，z=1.5 m 代表释放高度
Q_PUFF = 50.0                    # 单个 puff 的相对单位

# --- Wind（对应 FGP.WindSeries）
WIND_MODE = "randomwalk"      # "randomwalk"
U_MEAN = 2.0           # 平均风速 (m/s)
DIR0_DEG = 0.0         # 基准风向（度）0=朝 +x；90=朝 +y

# 正弦摆动参数（仅当 WIND_MODE="sin"）
VEER_AMP_DEG = 25.0    # 风向摆幅（±度数）
VEER_PERIOD_S = 60.0   # 摆动周期（秒）

# 随机游走参数（仅当 WIND_MODE="randomwalk"）
RW_SIGMA_DEG_PER_S = 1.2    # 每秒风向增量的标准差（度/秒）
RW_MAX_AMP_DEG = 45.0       # 最大允许偏移（度），防止漂移失控

# --- Diffusion: 横向/竖向扩散系数（sigma ~ a*sqrt(t)）
A_Y = 1.2  # 横向扩散系数（影响 plume 在 y 方向的扩展速度）
A_Z = 0.8  # 竖向扩散系数（含地面镜像；影响 near-ground 浓度）

# --- TimeParams: 仿真（对应 FGP.sim_dt / puff_dt / output_dt）
SIM_DT  = 1.0         # 仿真步长（s）（FGP: sim_dt）
PUFF_DT = 5.0         # puff 释放间隔（s）（FGP: puff_dt）
TOTAL_T = 6 * 60.0    # 总时长（s）
OUT_EVERY = 5         # 每隔多少步输出一帧（ FGP: output_dt = OUT_EVERY * SIM_DT）
GIF_NAME = "puff_wind_randomwalk.gif"

# --- Plot scaling
PERC_FOR_VMAX = 99.5  # 取 99.5% 分位作为 vmax，增强对比度

# 2) 网格构造（FGP: Geometry.grid） 
# 生成 2D 网格（X, Y），Z 为常数切片 后续在该切片上计算浓度并绘制热图。
xs = np.arange(Xmin, Xmax + dx, dx)
ys = np.arange(Ymin, Ymax + dy, dy)
X, Y = np.meshgrid(xs, ys, indexing="xy")  # X/Y shape 相同
Z = np.full_like(X, z_slice)               # Z 与 X/Y 同形状，值全为 z_slice

# 3) 风场时间函数（FGP: WindSeries） ==================
# FGP 要求“等间隔风观测序列”，并在内部插值到 sim_dt；这里我们直接“显式函数”生成 (Ux, Uy)。

def _wind_dir_deg_sin(t: float) -> float:
    """正弦摆动的风向（度）。FGP 中可看作给定方向序列的一个生成器。"""
    return DIR0_DEG + VEER_AMP_DEG * math.sin(2.0 * math.pi * (t / VEER_PERIOD_S))

_rw_dir_offset_deg = 0.0  # 随机游走的累积偏移

def _wind_dir_deg_randomwalk_current() -> float:
    """返回当前随机游走偏移叠加后的风向（度）。"""
    return DIR0_DEG + _rw_dir_offset_deg

def wind_vec_at(t: float) -> tuple[float, float]:
    """
    返回给定时刻的风矢量 (Ux, Uy)。
    Concept mapping: FGP.WindSeries.interpolate(t) → (U_x(t), U_y(t))
    """
    if WIND_MODE == "sin":
        th_deg = _wind_dir_deg_sin(t)
    elif WIND_MODE == "randomwalk":
        th_deg = _wind_dir_deg_randomwalk_current()
    else:
        th_deg = DIR0_DEG  # 固定风向
    th = math.radians(th_deg)
    Ux = U_MEAN * math.cos(th)
    Uy = U_MEAN * math.sin(th)
    return Ux, Uy

# 4) Puff 状态容器（FGP: puff 列表，c++部分） ==================
# 每个 puff: [x, y, z, age]
puffs: list[list[float]] = []
t_since_emit = PUFF_DT  # 使得启动时立即满足释放阈值

# 5) 物理核：单个 puff 的浓度贡献（FGP: C++ kernel / Gaussian solution） ==================
def cp_concentration(x: np.ndarray, y: np.ndarray, z: np.ndarray,
                     puff: list[float], Q: float) -> np.ndarray:
    """
    计算单个 puff 在网格点 (x, y, z) 的浓度贡献（含地面镜像法）。
    Concept mapping: 对应 FGP 的“单 puff 闭式解”，在 C++ 内用高效实现并批量化/裁剪。
    公式（简化形态）：
      c = Q / ((2π)^(3/2) σy^2 σz)
          * exp(-((x-xp)^2+(y-yp)^2)/(2σy^2))
          * [exp(-(z-zp)^2/(2σz^2)) + exp(-(z+zp)^2/(2σz^2))]
    其中 σy, σz ~ a*sqrt(age)；a 由大气稳定度等经验给出（这里用常数示意）。
    """
    xp, yp, zp, age = puff
    if age <= 0.0:
        # 初始 age 为 1e-6，避免 sqrt(0)；age<=0 时直接返回 0 场
        return np.zeros_like(x, dtype=np.float64)

    # 扩散尺度（随时间~sqrt(age) 增长；FGP 里会根据稳定度/参数化更细致）
    sig_y = max(1e-6, A_Y * math.sqrt(age))
    sig_z = max(1e-6, A_Z * math.sqrt(age))

    # 归一化系数
    norm = Q / ((2.0 * math.pi) ** 1.5 * (sig_y ** 2) * sig_z)

    # 水平距离平方
    rx2 = (x - xp) ** 2 + (y - yp) ** 2
    horiz = np.exp(-0.5 * rx2 / (sig_y ** 2))

    # 垂直方向采用“地面镜像”处理
    vert = (
        np.exp(-0.5 * ((z - zp) ** 2) / (sig_z ** 2)) +
        np.exp(-0.5 * ((z + zp) ** 2) / (sig_z ** 2))
    )

    return norm * horiz * vert

# 6) Puff 平流与老化（FGP: kernel 内对 puff 状态推进）
def advect_and_age(puffs_list: list[list[float]], dt: float, Ux: float, Uy: float) -> None:
    """
    用欧拉显式步进：位置 += U * dt；age += dt。
    Concept mapping: FGP 在 C++ 内部对所有 puff 做同样的推进，并在需要时做阈值化/裁剪。
    """
    for p in puffs_list:
        p[0] += Ux * dt  # x 平流
        p[1] += Uy * dt  # y 平流
        p[3] += dt       # age 增长

# 7) 单帧渲染（FGP: demo绘图： 这里用 Agg+动态vmax修改）
def render_frame(C: np.ndarray, t: float, npuffs: int, Ux: float, Uy: float) -> np.ndarray:
    """
    把当前切片浓度场 C 绘制成一帧图像，并返回 RGB ndarray（用于 GIF 拼接）。
    - 动态设定 vmax（99.5% 分位）增强对比度。
    - 左下角画风矢量，展示当前风向/风速。
    """

    vmax = np.nanpercentile(C, PERC_FOR_VMAX)
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1e-6
    vmin = 0.0

    fig, ax = plt.subplots(figsize=(7.5, 4.0), dpi=120)
    im = ax.imshow(
        C, origin="lower",
        extent=[Xmin, Xmax, Ymin, Ymax],
        interpolation="nearest",
        vmin=vmin, vmax=vmax
    )
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("concentration (arb. unit)")

    spd = math.hypot(Ux, Uy)  # 当前风速大小
    ax.set_title(
        f"Time-varying wind  z={z_slice} m | t={int(t)} s | puffs={npuffs} | "
        f"U=({Ux:.2f},{Uy:.2f}) m/s | maxC={C.max():.3e}"
    )
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.plot([SRC[0]], [SRC[1]], marker="*", markersize=10, color="white")  # 源位置标记
    # 画风矢量
    ax.quiver(Xmin + 20, Ymin + 20, Ux, Uy, angles='xy', scale_units='xy', scale=1, width=0.004)

    fig.tight_layout()
    # Matplotlib 3.9+：从渲染器取 RGBA 缓冲区，再裁到 RGB
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    frame = np.asarray(renderer.buffer_rgba())[:, :, :3]
    plt.close(fig)
    return frame

#  8) 主循环（FGP: solver.run() 的 Python 部分） ==================
def simulate() -> str:
    """
    核心流程：
      for t in 0..TOTAL_T step SIM_DT:
        - 更新风（FGP: WindSeries.interpolate(t)）
        - 根据 PUFF_DT 决定是否释放新 puff（FGP: emission schedule）
        - 平流 + 老化（FGP: 内核推进）
        - 域外裁剪（FGP: 阈值/边界裁剪的简单替代）
        - 叠加所有 puff 的浓度到网格（FGP: C++ 矢量化核的 Python 模拟版）
        - 按 OUT_EVERY 输出一帧
    返回：GIF
    """
    global t_since_emit, puffs, _rw_dir_offset_deg

    n_steps = int(TOTAL_T / SIM_DT)
    frames: list[np.ndarray] = []

    for step in range(n_steps):
        t = step * SIM_DT

        #  随机游走模式：风向在每个仿真步做一次“小步随机增量”
        if WIND_MODE == "randomwalk":
            _rw_dir_offset_deg += float(np.random.normal(0.0, RW_SIGMA_DEG_PER_S)) * float(SIM_DT)
            # 防止漂移过大：夹在 ±RW_MAX_AMP_DEG 内
            _rw_dir_offset_deg = float(np.clip(_rw_dir_offset_deg, -RW_MAX_AMP_DEG, RW_MAX_AMP_DEG))

        #  查询/生成当前风矢量（与 FGP.WindSeries.interpolate(t) 的角色相当）
        Ux, Uy = wind_vec_at(t)

        #  根据 puff_dt 释放 puff（与 FGP.Emission 定时生成 puff 的调度一致）
        t_since_emit += SIM_DT
        if t_since_emit >= PUFF_DT:
            puffs.append([float(SRC[0]), float(SRC[1]), float(SRC[2]), 1e-6])
            t_since_emit = 0.0

        #  平流 + 老化
        advect_and_age(puffs, SIM_DT, Ux=Ux, Uy=Uy)

        
        alive: list[list[float]] = []
        for p in puffs:
            xp, yp, *_ = p
            if Xmin - 50 <= xp <= Xmax + 50 and Ymin - 50 <= yp <= Ymax + 50:
                alive.append(p)
        puffs = alive

        #  叠加所有 puff 的贡献
        C = np.zeros_like(X, dtype=np.float64)
        for p in puffs:
            C += cp_concentration(X, Y, Z, p, Q_PUFF)

        #  采样输出（~ FGP.output_dt）
        if step % OUT_EVERY == 0:
            frames.append(render_frame(C, t, len(puffs), Ux, Uy))

    #  导出 GIF
    out_path = os.path.abspath(GIF_NAME)
    imageio.mimsave(out_path, frames, duration=0.12)
    return out_path

if __name__ == "__main__":
    print("[MAIN] Start mini puff with time-varying wind (annotated, FGP-mapped).")
    try:
        out_file = simulate()
        print(f"[OK] GIF saved: {out_file}")
    except Exception as e:
        print(f"[ERROR] {e!r}")
        raise
    finally:
        print("[MAIN] Exit.")
