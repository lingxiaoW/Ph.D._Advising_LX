"""
while True:
    c, (wind_v, wind_phi) = SensorReading(Drone_Position)       # Gas Model
    drone_status = Get_Drone_status(...)                        
    Next_Way_Point = Navigation_Controller(c, wind_phi, status) # Olfactory-based navigation
    cmd_way_point = drone_control(Next_Way_Point, status)       
    env.step(cmd_way_point)                                     
    if  drone_position == source_location:
        break


    - gasplume.py           ( load_puffs, puffs, wind_vec_at)
    - puffs_init.pkl        (预热好的plume)
    - sensor_reading.py     (sensor_reading)
    - get_drone_status.py   (bind_env_fetcher, get_drone_status)
    - navigation_controller.py (navigation_controller / Navigation_Controller)
    - drone_control.py      (drone_control / Drone_Control)
    - results.py            (write_csv, plot_3d, plot_timeseries)
"""

import math
from typing import Dict, List

import numpy as np

from gasplume import load_puffs, wind_vec_at
from sensor_reading import sensor_reading
from get_drone_status import bind_env_fetcher, get_drone_status
from navigation_controller import navigation_controller
from drone_control import drone_control
from results import write_csv, plot_3d, plot_timeseries


# ========= 参数 =========


DT: float = 0.5          # 仿真时间步长 [s]
MAX_T: float = 300.0     # 最大仿真时间 [s]
VMAX: float = 15.0       # 无人机最大速度 [m/s]

# 边界
XMIN, XMAX = -50.0, 300.0
YMIN, YMAX = -150.0, 150.0
ZMIN, ZMAX = 0.0, 20.0

# 源点 判定阈值
SRC = np.array([0.0, 0.0, 1.5], dtype=float)
SRC_TOL: float = 5.0     

# 无人机初始状态
START_POS = np.array([200.0, 0.0, 10.0], dtype=float)
START_HEADING_DEG: float = 0.0

# 导航参数（传给 navigation_controller）
C_THRESH: float = 1e-4   # 浓度阈值: c > C_THRESH ....
NAV_STEP: float = 5.0    # navigation_controller 每步给出的 waypoint 位移 

# 控制步长参数（传给 drone_control）
CTRL_STEP: float = 5.0 

# 输出
CSV_PATH = "run_olfactory.csv"
TRAJ_FIG = "traj_olfactory.png"
TEL_FIG = "telemetry_olfactory.png"

# ========= 


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def clamp_pos(pos: np.ndarray) -> np.ndarray:
    return np.array([
        clamp(pos[0], XMIN, XMAX),
        clamp(pos[1], YMIN, YMAX),
        clamp(pos[2], ZMIN, ZMAX),
    ], dtype=float)


def vec_norm(v: np.ndarray) -> float:
    return float(np.linalg.norm(v))


def main():
    print("[BOOT] main_olfactory starting...")

    #载入puff
    try:
        load_puffs("puffs_init.pkl")
        print("[PLUME] Loaded puffs_init.pkl for plume field.")
    except Exception as e:
        print(f"[PLUME][WARN] Failed to load puffs_init.pkl: {e}")
        print("              sensor_reading() will still run, but c may be ~0.")

    print(f"[WORLD] X=[{XMIN},{XMAX}] Y=[{YMIN},{YMAX}] Z=[{ZMIN},{ZMAX}]")
    print(f"[START] pos={START_POS}, SRC={SRC}, DT={DT}, VMAX={VMAX}, SRC_TOL={SRC_TOL}")
    print("[NAV] Using olfactory-based navigation: upwind if c>thresh, crosswind otherwise.")

    #初始化内部状态
    state: Dict[str, np.ndarray | float] = {
        "t": 0.0,
        "pos": START_POS.copy(),
        "vel": np.zeros(3, dtype=float),
        "heading_deg": START_HEADING_DEG,
    }

    # 提供给 get_drone_status 的回调
    def _fetch_state():
        return {
            "pos": state["pos"],
            "vel": state["vel"],
        }

    bind_env_fetcher(_fetch_state)

    # main loop
    rows: List[Dict[str, float]] = []
    max_c = 0.0

    while True:
        t = float(state["t"])
        pos = state["pos"]

        # 停止条件：时间
        if t > MAX_T:
            print("[STOP] Max flight time reached.")
            break

        # get_drone_status
        pos_gd, heading_deg = get_drone_status(t)

        #气体和风
        c, wind_v, wind_phi = sensor_reading(
            pos_gd[0], pos_gd[1], pos_gd[2], t, degrees=True
        )
        max_c = max(max_c, float(c))

        #给出下一航点
        next_wp = navigation_controller(
            c,
            wind_phi,
            {"pos": pos_gd},
            C_THRESH,
            NAV_STEP,
        )

        # 无人机控制
        cmd_wp = drone_control(
            next_wp,
            {"pos": pos_gd},
            CTRL_STEP,
        )

        # 计算速度
        step_vec = cmd_wp - pos
        dist = vec_norm(step_vec)
        if dist > 1e-9:
            direction = step_vec / dist
        else:
            direction = np.zeros(3, dtype=float)

        desired_speed = dist / DT if DT > 0 else 0.0
        speed = min(desired_speed, VMAX)

        vel = direction * speed
        new_pos = clamp_pos(pos + vel * DT)

        # 更新 heading
        if vec_norm(vel[:2]) > 1e-6:
            heading_deg = math.degrees(math.atan2(vel[1], vel[0]))

        state["t"] = t + DT
        state["pos"] = new_pos
        state["vel"] = vel
        state["heading_deg"] = heading_deg

        dist_to_src = vec_norm(SRC - new_pos)

        # 记录数据
        rows.append({
            "t": t,
            "x": pos[0],
            "y": pos[1],
            "z": pos[2],
            "speed": speed,
            "heading_deg": heading_deg,
            "c": float(c),
            "wind_v": float(wind_v),
            "wind_phi": float(wind_phi),
            "dist_to_src": dist_to_src,
        })

        
        if int(t / 5.0) != int((t - DT) / 5.0):
            print(
                f"[RUN] t={t:6.1f}s pos={pos} dist={dist_to_src:6.2f} "
                f"c={c:6.3e} wind=({wind_v:4.2f}m/s,{wind_phi:6.1f}°)"
            )

        # 命中判定
        if dist_to_src <= SRC_TOL and c > 0.0:
            print(
                f"[HIT] t={t:6.2f}s dist={dist_to_src:5.2f} "
                f"c={c:6.3e} (Reached source neighborhood)"
            )
            break

    # 结果
    if rows:
        write_csv(rows, CSV_PATH)
        print(f"[SAVE] CSV -> {CSV_PATH}; max_c={max_c:.3e}")

        path_xyz = [(r["x"], r["y"], r["z"]) for r in rows]
        c_list = [r["c"] for r in rows]
        bounds = (XMIN, XMAX, YMIN, YMAX, ZMIN, ZMAX)
        # 用最终风向画个参考
        Ux, Uy = wind_vec_at(state["t"])
        plot_3d(
            path_xyz,
            src_list=[SRC],
            bounds=bounds,
            path=TRAJ_FIG,
            c_list=c_list,
            wind_vec=(Ux, Uy),
        )
        print(f"[SAVE] Trajectory -> {TRAJ_FIG}")

        t_list = [r["t"] for r in rows]
        series = {
            "speed":       [r["speed"] for r in rows],
            "heading_deg": [r["heading_deg"] for r in rows],
            "c":           [r["c"] for r in rows],
            "wind_v":      [r["wind_v"] for r in rows],
        }
        plot_timeseries(t_list, series, out_png=TEL_FIG)
        print(f"[SAVE] Telemetry -> {TEL_FIG}")
    else:
        print("[WARN] No data recorded; nothing to save/plot.")

    print("[END] main_olfactory")


if __name__ == "__main__":
    main()
