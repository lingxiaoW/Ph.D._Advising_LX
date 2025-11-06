"""
drone_control.py
作用：
  1) 将 heading 指令转换为下一航点（waypoint）
  2) 将下一航点转换为速度指令（drone_speed_cmd）

角度约定：
  0° = +x 方向；角度逆时针为正（与 wind_phi / heading_deg 一致）

主要接口：
  next_wp = heading_to_waypoint(heading_deg, pos, step=5.0, hold_z=None, bounds=None)
  v_cmd   = drone_control(next_wp, current_pos, kp=0.8, vmax=22.0, vmin_kick=1.0)
"""

from __future__ import annotations
from typing import Optional, Tuple
import numpy as np
import math

__all__ = ["heading_to_waypoint","drone_control",]

# -------------------- Utils --------------------
def _norm_deg(a: float) -> float:
    a = float(a) % 360.0
    return a if a >= 0.0 else a + 360.0

def _heading_to_unit_vec(heading_deg: float) -> np.ndarray:
    """0°=+x，逆时针为正，仅在水平面产生单位向量"""
    r = math.radians(heading_deg)
    return np.array([math.cos(r), math.sin(r), 0.0], dtype=float)

def _clamp_bounds(p: np.ndarray, bounds: Optional[Tuple[float,float,float,float,float,float]]) -> np.ndarray:
    if bounds is None: return p
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    x = min(max(p[0], xmin), xmax)
    y = min(max(p[1], ymin), ymax)
    z = min(max(p[2], zmin), zmax)
    return np.array([x, y, z], dtype=float)

def _vec_norm(v: np.ndarray) -> float:
    return float(np.linalg.norm(v))

# ---------------- Heading → Waypoint ----------------
def heading_to_waypoint(
    heading_deg: float,
    pos: np.ndarray,
    *,
    step: float = 5.0,
    hold_z: Optional[float] = None,
    bounds: Optional[Tuple[float,float,float,float,float,float]] = None,
) -> np.ndarray:
    """
    输入：
      heading_deg : 目标朝向（度）
      pos         : 当前坐标 (x,y,z)
      step        : 沿 heading 前进的距离（米）
      hold_z      : 固定高度；None=保持当前 z
      bounds      : (xmin, xmax, ymin, ymax, zmin, zmax)
    输出：
      next_wp     : 下一航点 (x,y,z)
    """
    heading_deg = _norm_deg(float(heading_deg))
    dir_xy = _heading_to_unit_vec(heading_deg)
    next_wp = np.asarray(pos, dtype=float) + float(step) * dir_xy
    next_wp[2] = float(pos[2] if (hold_z is None) else hold_z)
    return _clamp_bounds(next_wp, bounds)

# ---------------- Waypoint → SpeedCmd ---------------
def drone_control(
    next_way_point: np.ndarray,
    current_pos: np.ndarray,
    *,
    kp: float = 0.8,
    vmax: float = 22.0,
    vmin_kick: float = 1.0,
) -> np.ndarray:
    """
    输入：
      next_way_point : 目标航点 (x,y,z)
      current_pos    : 当前坐标 (x,y,z)
      kp             : 比例系数（位置误差→速度）
      vmax           : 速度上限（m/s）
      vmin_kick      : 低速下限（避免停滞）
    输出：
      v_cmd          : 速度指令向量 (vx,vy,vz)
    """
    next_way_point = np.asarray(next_way_point, dtype=float)
    current_pos = np.asarray(current_pos, dtype=float)
    err = next_way_point - current_pos
    d = _vec_norm(err)
    if d <= 1e-9:
        return np.zeros(3, dtype=float)

    # 比例器 + 限幅
    v = kp * (err / d) * min(d, vmax)
    spd = _vec_norm(v)
    if spd > vmax:
        v = v * (vmax / (spd + 1e-12))
    if spd < vmin_kick:
        v = (err / d) * vmin_kick
    return v
