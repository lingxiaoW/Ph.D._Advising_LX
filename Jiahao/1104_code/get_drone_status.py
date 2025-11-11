
"""
Get Drone Status 
功能：return无人机位置 pos 与朝向 heading_deg（度）。
"""
from __future__ import annotations
import math
from typing import Callable, Dict
import numpy as np

_FETCH: Callable[[], Dict[str, np.ndarray]] | None = None
_LAST_HEADING: float = 0.0
_NEAR_ZERO = 1e-3  # m/s


def bind_env_fetcher(fetch_state: Callable[[], Dict[str, np.ndarray]]) -> None:
    """return {"pos": ndarray(3), "vel": ndarray(3)}。"""
    global _FETCH
    _FETCH = fetch_state

##由速度向量计算航向角（度）。atan2(y,x)，x轴为 0°，逆时针为正。
def _heading_deg(vx: float, vy: float) -> float:
    return math.degrees(math.atan2(vy, vx))

'''
    input：
        t 
    output:
        (pos, heading_deg)
        - pos: np.ndarray(3,) -> [x, y, z]
        - heading_deg: float -> （degrees）
'''

def get_drone_status(t: float):  
    """读取当前 pos 与 heading_deg"""
    global _LAST_HEADING
    if _FETCH is None:
        raise RuntimeError("bind_env_fetcher(...) must be called before get_drone_status().")
    st = _FETCH()
    pos = np.asarray(st.get("pos", (0.0, 0.0, 0.0)), dtype=float)
    vel = np.asarray(st.get("vel", (0.0, 0.0, 0.0)), dtype=float)
    if float(np.linalg.norm(vel)) < _NEAR_ZERO:
        hd = _LAST_HEADING
    else:
        hd = _heading_deg(float(vel[0]), float(vel[1]))
        _LAST_HEADING = hd
    
    #output drone_position, drone_heading (pos,hd)
    return pos, hd
