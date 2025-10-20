
"""
主循环与计算
- 随机游走的风向模型
"""

import math
import os
from typing import List, Tuple

import numpy as np

import params_randomwalk as P
import viz_output as viz


#  Geometry
def build_grid() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs = np.arange(P.Xmin, P.Xmax + P.dx, P.dx)
    ys = np.arange(P.Ymin, P.Ymax + P.dy, P.dy)
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    Z = np.full_like(X, P.z_slice, dtype=float)
    return X, Y, Z


# 随机游走风向状态
_rw_dir_offset_deg = 0.0  # 累计偏移角


def _advance_randomwalk(dt: float) -> None:
    """推进随机游走：每步给风向增加一个高斯增量，并做限幅。"""
    global _rw_dir_offset_deg
    _rw_dir_offset_deg += float(np.random.normal(0.0, P.RW_SIGMA_DEG_PER_S)) * float(dt)
    _rw_dir_offset_deg = float(np.clip(_rw_dir_offset_deg, -P.RW_MAX_AMP_DEG, P.RW_MAX_AMP_DEG))


def wind_vec_randomwalk() -> Tuple[float, float]:
    """把当前风向角（基准 + 偏移）转换成 (Ux, Uy)。"""
    th = math.radians(P.DIR0_DEG + _rw_dir_offset_deg)
    Ux = P.U_MEAN * math.cos(th)
    Uy = P.U_MEAN * math.sin(th)
    return Ux, Uy


# 单个 puff 的浓度 
def cp_concentration(x: np.ndarray, y: np.ndarray, z: np.ndarray,
                     puff: List[float], Q: float) -> np.ndarray:
    """
    单 puff 的 3D 高斯贡献（含地面镜像）：
      c = Q / ((2π)^(3/2) σy^2 σz) * exp(-((x-xp)^2+(y-yp)^2)/(2σy^2))
          * [exp(-(z-zp)^2/(2σz^2)) + exp(-(z+zp)^2/(2σz^2))]
    其中 σy, σz ~ a*sqrt(age)
    """
    xp, yp, zp, age = puff
    if age <= 0.0:
        return np.zeros_like(x, dtype=np.float64)

    sig_y = max(1e-6, P.A_Y * math.sqrt(age))
    sig_z = max(1e-6, P.A_Z * math.sqrt(age))
    norm = Q / ((2.0 * math.pi) ** 1.5 * (sig_y ** 2) * sig_z)

    rxy2 = (x - xp) ** 2 + (y - yp) ** 2
    horiz = np.exp(-0.5 * rxy2 / (sig_y ** 2))
    vert = (np.exp(-0.5 * ((z - zp) ** 2) / (sig_z ** 2))
            + np.exp(-0.5 * ((z + zp) ** 2) / (sig_z ** 2)))
    return norm * horiz * vert


#puff 推进 
def advect_and_age(puffs: List[List[float]], dt: float, Ux: float, Uy: float) -> None:
    """欧拉步进：位置 += U*dt；age += dt"""
    for p in puffs:
        p[0] += Ux * dt
        p[1] += Uy * dt
        p[3] += dt


#主循环
def simulate() -> str:
    # 网格
    X, Y, Z = build_grid()

    # 每个 puff = [x, y, z, age]
    puffs: List[List[float]] = []

    # 让启动时立刻释放puff
    t_since_emit = P.PUFF_DT

    frames: List[np.ndarray] = []

    n_steps = int(P.TOTAL_T / P.SIM_DT)
    for step in range(n_steps):
        t = step * P.SIM_DT

        # 随机游走：推进一个小的方向增量
        _advance_randomwalk(P.SIM_DT)

        # 当前风矢量
        Ux, Uy = wind_vec_randomwalk()

        # 定时释放 puff
        t_since_emit += P.SIM_DT
        if t_since_emit >= P.PUFF_DT:
            px, py, pz = P.SRC
            puffs.append([float(px), float(py), float(pz), 1e-6])  # age 从极小正数开始
            t_since_emit = 0.0

        # 平流
        advect_and_age(puffs, P.SIM_DT, Ux, Uy)

        # 边界裁剪 防止列表无限增长
        alive: List[List[float]] = []
        for p in puffs:
            xp, yp, *_ = p
            if P.Xmin - 50 <= xp <= P.Xmax + 50 and P.Ymin - 50 <= yp <= P.Ymax + 50:
                alive.append(p)
        puffs = alive

        # 叠加所有 puff 的浓度贡献
        C = np.zeros_like(X, dtype=np.float64)
        for p in puffs:
            C += cp_concentration(X, Y, Z, p, P.Q_PUFF)

        # 输出（每 OUT_EVERY 步渲染一帧）
        if step % P.OUT_EVERY == 0:
            frame = viz.render_frame(C, t, len(puffs), Ux, Uy, X, Y)
            frames.append(frame)

    
    out_path = viz.save_gif(frames, P.GIF_NAME)
    return out_path


if __name__ == "__main__":
    print("[MAIN] Start randomwalk puff simulation...")
    try:
        out_file = simulate()
        print(f"[OK] GIF saved: {os.path.abspath(out_file)}")
    except Exception as e:
        print(f"[ERROR] {e!r}")
        raise
    finally:
        print("[MAIN] Exit.")
