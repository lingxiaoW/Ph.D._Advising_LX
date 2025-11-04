# puff_plume.py
# -*- coding: utf-8 -*-
"""
PuffPlume: 从 gasplume_puff_3d.py 提炼的可调用羽流类
提供:
  - step(dt): 推进内部时间，按 SIM_DT 节拍做对流、老化、发射
  - sample(x, y, z): 采样当前时刻指定点的浓度
  - slice2d(z, dx=None, dy=None): 计算给定 z 的二维浓度切片 (X, Y, C)
说明:
  - 单位与原脚本一致：米、秒
  - 风场：默认 (U_MEAN, 0)，可自定义 wind_vec_at()
"""

from __future__ import annotations
import math
import numpy as np
from typing import List, Tuple

# ==== 域范围（与原脚本一致/兼容） ====
Xmin, Xmax = -50.0, 300.0
Ymin, Ymax = -150.0, 150.0

# 网格缺省分辨率（仅用于 slice2d）
DX = DY = 2.0

# 源与排放/风参数（与原脚本一致/兼容）
SRC = np.array([0.0, 0.0, 1.5], dtype=float)
Q_PUFF = 50.0       # 单个 puff 的标称强度（任意单位）
U_MEAN  = 2.0       # 平均风速（x方向）
A_Y = 1.2           # 横向扩散系数
A_Z = 0.8           # 垂向扩散系数

# 模拟节拍（保持与你的脚本一致）
SIM_DT  = 1.0       # plume 物理推进步长
PUFF_DT = 5.0       # 发射间隔

# 可选：百分位上限用于可视化归一（不影响数值采样）
PERC_FOR_VMAX = 99.5


def default_wind_vec_at(t: float) -> Tuple[float, float]:
    """默认风场：恒定 (U_MEAN, 0) ；如需时变，可按需改写。"""
    return float(U_MEAN), 0.0


def cp_concentration(x, y, z, puff, Q):
    """
    单个高斯 puff 的浓度贡献（与你原脚本同式）
    puff = (xp, yp, zp, age)
    """
    xp, yp, zp, age = puff
    if age <= 0.0:
        return np.zeros_like(x, dtype=np.float64)
    sig_y = max(1e-6, A_Y * math.sqrt(age))
    sig_z = max(1e-6, A_Z * math.sqrt(age))
    norm = Q / ((2.0 * math.pi) ** 1.5 * (sig_y ** 2) * sig_z)
    rx2  = (x - xp) ** 2 + (y - yp) ** 2
    horiz = np.exp(-0.5 * rx2 / (sig_y ** 2))
    # 含地面镜像
    vert  = np.exp(-0.5 * ((z - zp) ** 2) / (sig_z ** 2)) \
          + np.exp(-0.5 * ((z + zp) ** 2) / (sig_z ** 2))
    return norm * horiz * vert


def advect_and_age(puffs_list, dt, Ux, Uy):
    """对流 + 老化（与你原脚本同式）"""
    for p in puffs_list:
        p[0] += Ux * dt
        p[1] += Uy * dt
        p[3] += dt


class PuffPlume:
    def __init__(self,
                 wind_fn=default_wind_vec_at,
                 xlim=(Xmin, Xmax),
                 ylim=(Ymin, Ymax),
                 src=SRC,
                 q_puff=Q_PUFF,
                 sim_dt=SIM_DT,
                 puff_dt=PUFF_DT):
        self.wind_fn  = wind_fn
        self.xlim     = xlim
        self.ylim     = ylim
        self.src      = np.array(src, dtype=float).reshape(3)
        self.q_puff   = float(q_puff)
        self.sim_dt   = float(sim_dt)
        self.puff_dt  = float(puff_dt)

        self.puffs: List[List[float]] = []   # 每个元素 [xp, yp, zp, age]
        self._accum = 0.0                    # 与 sim_dt 对齐的内部累积时钟
        self._since_emit = self.puff_dt      # 控制发射节拍
        self.t = 0.0

    def _emit_if_needed(self, step_dt, Ux, Uy):
        self._since_emit += step_dt
        if self._since_emit >= self.puff_dt:
            # 新 puff，从源位置开始，给一个极小 age 以避免 sig=0
            self.puffs.append([float(self.src[0]), float(self.src[1]), float(self.src[2]), 1e-6])
            self._since_emit = 0.0

    def _clip_alive(self):
        x0, x1 = self.xlim
        y0, y1 = self.ylim
        margin = 50.0
        self.puffs = [p for p in self.puffs
                      if (x0 - margin <= p[0] <= x1 + margin) and (y0 - margin <= p[1] <= y1 + margin)]

    def _do_one_physics_tick(self):
        """以 sim_dt 做一次真实物理推进"""
        Ux, Uy = self.wind_fn(self.t)
        self._emit_if_needed(self.sim_dt, Ux, Uy)
        advect_and_age(self.puffs, self.sim_dt, Ux, Uy)
        self._clip_alive()
        self.t += self.sim_dt

    def step(self, dt: float):
        """
        外部以任意 dt 调用；内部在累积到 sim_dt 时才做一次真实推进。
        这样无人机可以 0.01s 控制，而羽流每 1s 推进一次。
        """
        if dt <= 0:
            return
        self._accum += dt
        while self._accum >= self.sim_dt:
            self._do_one_physics_tick()
            self._accum -= self.sim_dt

    def sample(self, x: float, y: float, z: float) -> float:
        """采样当前时刻 (x,y,z) 的浓度（所有 puff 线性叠加）"""
        if len(self.puffs) == 0:
            return 0.0
        # 向量化计算：对每个 puff 叠加
        # 这里为了简洁，直接循环累加；若性能不足，可做邻域过滤或向量化堆叠
        c = 0.0
        for p in self.puffs:
            c += float(cp_concentration(np.array([x]), np.array([y]), np.array([z]), p, self.q_puff)[0])
        return c

    def slice2d(self, z: float, dx: float | None = None, dy: float | None = None):
        """
        返回 (X, Y, C) 网格（用于等值线/等值填色可视化），不做绘图。
        """
        if dx is None: dx = DX
        if dy is None: dy = DY
        xs = np.arange(self.xlim[0], self.xlim[1] + dx, dx, dtype=float)
        ys = np.arange(self.ylim[0], self.ylim[1] + dy, dy, dtype=float)
        X, Y = np.meshgrid(xs, ys, indexing="xy")
        Z = np.full_like(X, float(z))
        C = np.zeros_like(X, dtype=np.float64)
        if len(self.puffs) > 0:
            for p in self.puffs:
                C += cp_concentration(X, Y, Z, p, self.q_puff)
        return X, Y, C
