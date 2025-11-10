"""
navigation_controller.py
Input:
    c           : gas concentration at UAV location
    phi         : wind direction angle (deg), 0° = +X, CCW positive
    drone_state : from Get_Drone_status()
        
    If c > C_THRESH:
        target_heading = phi + 180°
    Else:
        target_heading = phi + 90°

Output:
    next_way_point : np.ndarray([x_next, y_next, z_next])
                    
"""

import math
import numpy as np


def navigation_controller(
    c: float,
    wind_phi_deg: float,
    drone_state: dict,
    c_thresh: float,
    step: float,) -> np.ndarray:

    pos = np.array(drone_state["pos"], dtype=float)

    #浓度
    if c > c_thresh:
        heading = wind_phi_deg + 180.0
    else:
        heading = wind_phi_deg + 90.0

    #角度归一化
    heading = (heading + 180.0) % 360.0 - 180.0

    #根据 heading & step 计算 waypoint
    rad = math.radians(heading)
    dx = step * math.cos(rad)
    dy = step * math.sin(rad)

    next_wp = pos + np.array([dx, dy, 0.0], dtype=float)
    return next_wp

