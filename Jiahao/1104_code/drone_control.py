"""
drone_control.py

Input
    heading_command:
       drone_state: {"pos": np.array([x, y, z])}
    next_way_point: np.array([x, y, z])

Output:
    way_point: np.array([x, y, z]) 
"""

import math
import numpy as np


def drone_control(
    target,
    drone_state: dict,
    step: float,) -> np.ndarray:

    pos = np.array(drone_state["pos"], dtype=float)

    #已经是waypoint
    if isinstance(target, (list, tuple, np.ndarray)) and len(target) == 3:
        return np.array(target, dtype=float)

    # 按航向角生成 waypoint
    heading_deg = float(target)
    # 归一化
    heading_deg = (heading_deg + 180.0) % 360.0 - 180.0
    rad = math.radians(heading_deg)

    dx = step * math.cos(rad)
    dy = step * math.sin(rad)

    return pos + np.array([dx, dy, 0.0], dtype=float)



