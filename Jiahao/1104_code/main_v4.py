"""
main_v4.py
模拟无人机飞行的执行文件
"""
from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np
import math

#参数
DT = 0.02
KP = 0.6         
VMAX = 10.0       
VMIN_KICK = 1.0
SRC_TOL = 2.0
MAX_T = 180.0

#边界
XMIN, XMAX = -50.0, 300.0
YMIN, YMAX = -150.0, 150.0
ZMIN, ZMAX = 0.0, 20.0
WORLD_BOUNDS: Tuple[float, float, float, float, float, float] = (XMIN, XMAX, YMIN, YMAX, ZMIN, ZMAX)

# 起点 和 烟雾源点
START_POS = np.array([200.0, 100.0, 10.0], dtype=float)
SRC       = np.array([0.0, 0.0, 1.5], dtype=float)

#输出文件
CSV_PATH  = "run_v4.csv"
PLOT_PATH = "traj_v4.png"
TS_PATH   = "telemetry_v4.png"

# import部分的外部module
from get_drone_status import bind_env_fetcher, get_drone_status
from sensor_reading import sensor_reading
from results import write_csv, plot_3d, plot_timeseries
from navigation_controller import Navigation_Controller  # 需支持 crosswind_mode / threshold / return_debug

#plume加载
USE_WARMUP_FILE = True
PUFFS_FILE = "puffs_init.pkl"
try:
    from gasplume import puffs as PLUME_PUFFS, wind_vec_at, load_puffs
    if USE_WARMUP_FILE:
        try:
            load_puffs(PUFFS_FILE)
        except FileNotFoundError:
            print(f"[WARN] {PUFFS_FILE} not found; run gasplume_xy_final.py first.")
except Exception as e:
    print(f"[WARN] plume import failed: {e}. Fallback to constant +x wind. ({e})")
    PLUME_PUFFS = []
    def wind_vec_at(t: float):
        return (2.0, 0.0)


def vec_norm(v: np.ndarray) -> float:
    return float(np.linalg.norm(v))

def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))

def capped_velocity(to_wp_vec: np.ndarray) -> np.ndarray:
    d = vec_norm(to_wp_vec)
    if d <= 1e-9:
        return np.zeros(3)
    v = KP * (to_wp_vec / max(d, 1e-12)) * min(d, VMAX)
    spd = vec_norm(v)
    if spd > VMAX:
        v *= (VMAX / (spd + 1e-12))
    if spd < VMIN_KICK:
        v = (to_wp_vec / max(d, 1e-12)) * VMIN_KICK
    return v

def env_step(state: Dict[str, np.ndarray], v_cmd: np.ndarray, dt: float) -> None:
    v_cmd = np.asarray(v_cmd, dtype=float)
    spd = vec_norm(v_cmd)
    if spd > VMAX:
        v_cmd *= (VMAX / (spd + 1e-12))
    state['x'][0] = state['x'][0] + v_cmd * dt
    state['v'][0] = v_cmd
    x, y, z = state['x'][0]
    x = clamp(x, XMIN, XMAX); y = clamp(y, YMIN, YMAX); z = clamp(z, ZMIN, ZMAX)
    state['x'][0] = np.array([x, y, z], dtype=float)

def clamp_wp(p: np.ndarray, bounds: Tuple[float, float, float, float, float, float]) -> np.ndarray:
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    return np.array([min(max(p[0], xmin), xmax),
                     min(max(p[1], ymin), ymax),
                     min(max(p[2], zmin), zmax)], dtype=float)

def heading_to_unit_vec(deg: float) -> np.ndarray:
    r = math.radians(deg % 360.0)
    return np.array([math.cos(r), math.sin(r), 0.0], dtype=float)

# 主程序
def main():
    print("[START] main_v4 (zig-zag + easier upwind)")
    print(f"[WORLD] X=[{XMIN},{XMAX}] Y=[{YMIN},{YMAX}] Z=[{ZMIN},{ZMAX}]")

    #初始状态
    state = {'x': np.zeros((1,3), dtype=float), 'v': np.zeros((1,3), dtype=float)}
    state['x'][0] = START_POS.copy()

    # get_drone_status
    def _fetch_state():
        return {"pos": state['x'][0].copy(), "vel": state['v'][0].copy()}
    bind_env_fetcher(_fetch_state)

    csv_rows: List[Dict[str, float]] = []
    wps_hist: List[tuple] = []
    t = 0.0
    max_c = 0.0
    THRESH_EST = 0.01  

    while True:
        pos = state['x'][0]
        err = SRC - pos
        dist = vec_norm(err)

        # 状态
        pos2, heading_deg = get_drone_status(t)
        speed = vec_norm(state['v'][0])

        # wind和烟雾传感器
        try:
            c, wind_v, wind_phi = sensor_reading(pos2[0], pos2[1], pos2[2], t)
        except Exception:
            Ux, Uy = wind_vec_at(t)
            c, wind_v, wind_phi = 0.0, float(np.hypot(Ux, Uy)), 0.0

        max_c = max(max_c, float(c))

        #导航控制器
        wind_info = {"phi": wind_phi}
        drone_status = {"pos": pos2, "heading": heading_deg, "speed": speed, "t": t}
        next_wp, dbg = Navigation_Controller(
            c, wind_info, drone_status,
            threshold=0.01,             
            crosswind_mode="alternate", # ±90°
            alternate_period=6.0,       
            step=20.0,                  
            hold_z=START_POS[2],        # 固定高度
            world_bounds=WORLD_BOUNDS,
            return_debug=True
        )

        
        next_wp = clamp_wp(next_wp, WORLD_BOUNDS)

        # 防卡死
        if np.allclose(next_wp, state['x'][0], atol=1e-6):
            if c > THRESH_EST:
                heading_cmd = (wind_phi + 180.0) % 360.0
                reason = "upwind (fallback)"
            else:
                heading_cmd = (wind_phi - 90.0) % 360.0
                reason = "crosswind-alt (fallback)"
            dir_alt = heading_to_unit_vec(heading_cmd)
            next_wp = state['x'][0] + 5.0 * dir_alt
            next_wp = clamp_wp(next_wp, WORLD_BOUNDS)
        else:
            heading_cmd = float(dbg.get("heading_cmd", (wind_phi + (180.0 if c > THRESH_EST else 90.0)) % 360.0))
            reason = dbg.get("reason", "n/a")

        # 记录waypoint
        wps_hist.append((float(next_wp[0]), float(next_wp[1]), float(next_wp[2])))

        # 执行一步
        to_wp = next_wp - state['x'][0]
        v_cmd = capped_velocity(to_wp)
        env_step(state, v_cmd, DT)

        # CSV
        csv_rows.append({
            't': round(t, 6),
            'x': float(pos2[0]), 'y': float(pos2[1]), 'z': float(pos2[2]),
            'speed': speed, 'heading_deg': heading_deg,
            'c': float(c), 'wind_v': float(wind_v), 'wind_phi': float(wind_phi),
            'next_wp_x': float(next_wp[0]), 'next_wp_y': float(next_wp[1]), 'next_wp_z': float(next_wp[2]),
            'heading_cmd': float(heading_cmd), 'reason': reason,
            'dist_to_src': dist, 'status': 'RUN',
        })

        # 终止条件
        if dist <= SRC_TOL:
            print("[HIT] Reach source neighborhood.")
            break
        t += DT
        if t >= MAX_T:
            print("[STOP] Max flight time reached.")
            break

    # 输出
    write_csv(CSV_PATH, csv_rows)
    print(f"[SAVE] CSV -> {CSV_PATH}; max_c={max_c:.3e}")

    # 轨迹
    path_xyz = [(r['x'], r['y'], r['z']) for r in csv_rows]
    c_list   = [r.get('c', float('nan')) for r in csv_rows]
    Ux, Uy = wind_vec_at(t)
    plot_3d(path_xyz, wps_hist, WORLD_BOUNDS, path=PLOT_PATH, c_list=c_list, wind_vec=(Ux,Uy))

    # 时间序列
    t_list = [r['t'] for r in csv_rows]
    series = {
        'speed':       [r['speed'] for r in csv_rows],
        'heading_deg': [r['heading_deg'] for r in csv_rows],
        'heading_cmd': [r['heading_cmd'] for r in csv_rows],
        'c':           [r['c'] for r in csv_rows],
        'wind_v':      [r['wind_v'] for r in csv_rows],
    }
    plot_timeseries(t_list, series, out_png=TS_PATH)

    print("[END] main_v4")

if __name__ == '__main__':
    main()
