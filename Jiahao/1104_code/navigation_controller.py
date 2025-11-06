"""
navigation_controller.py
  If c > threshold: target_heading = wind_dir + 180 
  Else:             target_heading = wind_dir + 90 

Inputs:
  c            : float, gas concentration
  wind_info    : dict with {"phi": wind_dir_deg}  # degrees, 0°=+x, CCW positive
  drone_status : dict with {"pos": np.ndarray(3,), "heading": float, "speed": float, "t": float}
Outputs:  heading_cmd  （heading command）
  next_wp      : np.ndarray(3,)  # next waypoint (x,y,z)
"""

from __future__ import annotations
from typing import Dict, Tuple, Optional
import numpy as np
import math


def _norm_deg(a: float) -> float:
    a = float(a) % 360.0
    return a if a >= 0.0 else a + 360.0

def _heading_to_unit_vec(heading_deg: float) -> np.ndarray:
    r = math.radians(heading_deg)
    return np.array([math.cos(r), math.sin(r), 0.0], dtype=float)

def _clamp_bounds(p: np.ndarray, bounds: Optional[Tuple[float,float,float,float,float,float]]) -> np.ndarray:
    if bounds is None:
        return p
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    x = min(max(p[0], xmin), xmax)
    y = min(max(p[1], ymin), ymax)
    z = min(max(p[2], zmin), zmax)
    return np.array([x, y, z], dtype=float)

_ALT_STATE = {"sign": 1, "last_flip_t": 0.0}


def Navigation_Controller(
    
    ###input: c and wind_info
    c: float,
    wind_info: Dict[str, float],
    drone_status: Dict[str, float | np.ndarray],
    *,
    threshold: float = 0.05,
    crosswind_mode: str = "fixed",      
    alternate_period: float = 3.0,      
    step: float = 5.0,                  
    hold_z: Optional[float] = None,     #保持当前高度
    world_bounds: Optional[Tuple[float,float,float,float,float,float]] = None,
    return_debug: bool = False,
):
    """
    Return:
      next_waypoint (np.ndarray shape (3,))
    """
    pos = np.asarray(drone_status.get("pos", np.zeros(3)), dtype=float)
    t   = float(drone_status.get("t", 0.0))
    wind_dir = float(wind_info.get("phi", 0.0))   # degrees
    
    ###output: heading_cmd  （heading command）
    if c > float(threshold):
        heading_cmd = _norm_deg(wind_dir + 180.0)  # upwind
        reason = "upwind (c>threshold)"
        sign_use = 0
    else:
        if crosswind_mode == "alternate":
            # 每隔 alternate_period 秒在 +90/-90 间切换一次
            if (t - _ALT_STATE["last_flip_t"]) >= float(alternate_period):
                _ALT_STATE["sign"] *= -1
                _ALT_STATE["last_flip_t"] = t
            sign = _ALT_STATE["sign"]  # ±1
        else:
            sign = +1  # 固定 +90°
        heading_cmd = _norm_deg(wind_dir + 90.0 * sign)
        reason = "crosswind (search)"
        sign_use = sign

    # 由 heading 生成下一个waypoint
    dir_xy = _heading_to_unit_vec(heading_cmd)
    next_wp = pos + step * dir_xy
    next_wp[2] = float(pos[2] if (hold_z is None) else hold_z)
    next_wp = _clamp_bounds(next_wp, world_bounds)

    if return_debug:
        return next_wp, {"heading_cmd": heading_cmd, "reason": reason, "sign": sign_use}
    return next_wp
