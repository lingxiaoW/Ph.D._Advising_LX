

import os, time, math
from typing import Dict, Optional, List, Tuple, Any
import numpy as np

try:
    import torch
    torch.set_default_dtype(torch.float64)
except Exception:
    torch = None

# ------------------------------
# RotorPy 环境
# ------------------------------
try:
    from rotorpy.learning.quadrotor_environments import QuadrotorEnv
except Exception as e:
    raise ImportError("未找到 RotorPy。请先安装 rotorpy 并确保可导入 QuadrotorEnv。") from e

# ------------------------------
# PuffPlume
# ------------------------------
from puff_plume import PuffPlume  # 确保同目录存在


# ==================================================
# 参数
# ==================================================

class C:
    # ==== 开关 ====
    USE_OLFACTORY_NAV      = True     # True: 嗅觉导航；False: 随机航点

    # ==== 控制/时间 ====
    DT_GLOBAL              = 0.01
    HOVER_AFTER_REACH      = 0.02
    VMIN_KICK              = 0.6
    KICK_DIST_FACTOR       = 3.0
    KP                     = 3.8
    VMAX                   = 12.0     

    # ==== 世界范围/随机点 ====
    WORLD_HALF_EXTENT      = 300.0
    WORLD_MARGIN           = 0.30
    NUM_RANDOM_WP          = 30
    Z_MIN, Z_MAX           = 0.8, 1.6
    LOCAL_R_STEP           = 60.0

    # ==== 嗅觉导航 ====
    DESIRED_ALT            = 1.5
    ALT_KP                 = 2.0
    CONC_THRESHOLD         = 1e-6    
    NAV_SPEED              = 8.0      


    XY_REACH_R             = 0.8     
    Z_REACH_R              = 0.3      
    REACH_NEED_CONC        = True
    MIN_OLF_TIME_SEC       = 120.0    

    # ==== 输出 ====
    OUTPUT_DIR             = "output"
    CSV_FILE               = os.path.join(OUTPUT_DIR, "flight_log.csv")
    FIG_TRAJ_3D            = os.path.join(OUTPUT_DIR, "trajectory_3d.png")
    FIG_TRAJ_XY            = os.path.join(OUTPUT_DIR, "trajectory_xy.png")
    FIG_ALT_TIME           = os.path.join(OUTPUT_DIR, "altitude_time.png")
    ANIM_3D_MP4            = os.path.join(OUTPUT_DIR, "trajectory_3d.mp4")
    ANIM_FPS               = 30
    ANIM_STRIDE            = 1
    ANIM_ELEV, ANIM_AZIM   = 25.0, -60.0

    # ==== 可选：XY+羽流等值叠加 ====
    OVERLAY_CONTOUR        = False
    FIG_XY_WITH_PLUME      = os.path.join(OUTPUT_DIR, "trajectory_xy_with_plume.png")


# ==================================================
# 输出与绘图
# ==================================================

def save_csv(rows: List[List[str]], csv_path: str) -> None:
    import csv
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "time_s","phase",
            "x","y","z","vx","vy","vz","speed",
            "wp_x","wp_y","wp_z","v_cmd_x","v_cmd_y","v_cmd_z",
            "conc"
        ])
        w.writerows(rows)
    print(f"[SAVE] 轨迹已保存：{os.path.abspath(csv_path)}")


def plot_static(xs: np.ndarray, ys: np.ndarray, zs: np.ndarray, waypoints: List[np.ndarray],
                dt: float, fig_traj_3d: str, fig_traj_xy: str, fig_alt_time: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except Exception as e:
        print("[WARN] 绘图跳过:", e); return

    os.makedirs(os.path.dirname(fig_traj_3d) or ".", exist_ok=True)

    # 3D
    fig = plt.figure(figsize=(6.08, 6.08), dpi=100)
    ax = fig.add_subplot(111, projection='3d')  # type: ignore
    ax.plot(xs, ys, zs, label='trajectory')
    if len(xs) > 0:
        ax.scatter(xs[0], ys[0], zs[0], marker='o', s=40, label='start')
        ax.scatter(xs[-1], ys[-1], zs[-1], marker='^', s=40, label='end')
    if waypoints:
        wps = np.stack(waypoints, 0)
        ax.scatter(wps[:,0], wps[:,1], wps[:,2], marker='x', s=50, label='waypoints')
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z'); ax.set_title('3D Trajectory')
    try: ax.set_box_aspect((1,1,1))  # type: ignore
    except Exception: pass
    ax.set_xlim(-C.WORLD_HALF_EXTENT, C.WORLD_HALF_EXTENT)
    ax.set_ylim(-C.WORLD_HALF_EXTENT, C.WORLD_HALF_EXTENT)
    ax.set_zlim(C.Z_MIN-0.5, max(C.Z_MAX, 20.0))
    ax.legend(); fig.tight_layout(); fig.savefig(fig_traj_3d, dpi=150); plt.close(fig)
    print(f"[SAVE] 3D 轨迹图：{fig_traj_3d}")

    # XY
    fig = plt.figure(figsize=(6.08, 6.08), dpi=100)
    ax = fig.add_subplot(111)
    ax.plot(xs, ys, label='trajectory')
    if len(xs) > 0:
        ax.scatter(xs[0], ys[0], marker='o', s=40, label='start')
        ax.scatter(xs[-1], ys[-1], marker='^', s=40, label='end')
    if waypoints:
        wps = np.stack(waypoints, 0)
        ax.scatter(wps[:,0], wps[:,1], s=50, marker='x', label='waypoints')
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_title('XY Top-Down')
    ax.set_xlim(-C.WORLD_HALF_EXTENT, C.WORLD_HALF_EXTENT)
    ax.set_ylim(-C.WORLD_HALF_EXTENT, C.WORLD_HALF_EXTENT)
    ax.axis('equal'); ax.grid(True); ax.legend()
    fig.tight_layout(); fig.savefig(fig_traj_xy, dpi=150); plt.close(fig)
    print(f"[SAVE] XY 俯视图：{fig_traj_xy}")

    # Altitude-Time
    if len(zs) > 0:
        t = np.arange(len(zs))*dt
        fig = plt.figure(figsize=(7,3.5)); ax = fig.add_subplot(111)
        ax.plot(t, zs); ax.set_xlabel('Time [s]'); ax.set_ylabel('Z [m]'); ax.grid(True)
        ax.set_title('Altitude over Time')
        fig.tight_layout(); fig.savefig(fig_alt_time, dpi=150); plt.close(fig)
        print(f"[SAVE] 高度-时间：{fig_alt_time}")


def render_3d_animation(xs: np.ndarray, ys: np.ndarray, zs: np.ndarray, waypoints: List[np.ndarray],
                        out_mp4: str, fps: int, stride: int, elev: float, azim: float) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        import imageio.v3 as iio
    except Exception as e:
        print("[WARN] 3D 动画跳过：", e); return

    os.makedirs(os.path.dirname(out_mp4) or ".", exist_ok=True)

    xlim = (-C.WORLD_HALF_EXTENT, C.WORLD_HALF_EXTENT)
    ylim = (-C.WORLD_HALF_EXTENT, C.WORLD_HALF_EXTENT)
    zlim = (C.Z_MIN-0.5, max(C.Z_MAX, 20.0))

    fig = plt.figure(figsize=(6.08, 6.08), dpi=100)
    ax = fig.add_subplot(111, projection='3d')  # type: ignore
    wps = np.stack(waypoints, 0) if waypoints else None

    frames: List[np.ndarray] = []
    idxs = list(range(1, len(xs), stride))
    if idxs and idxs[-1] != len(xs) - 1: idxs.append(len(xs) - 1)

    for idx in idxs:
        ax.clear()
        ax.plot(xs[:idx], ys[:idx], zs[:idx], label='trajectory')
        if len(xs) > 0:
            ax.scatter(xs[0], ys[0], zs[0], marker='o', s=40, label='start')
            ax.scatter(xs[idx-1], ys[idx-1], zs[idx-1], marker='^', s=40, label='current')
        if wps is not None:
            ax.scatter(wps[:,0], wps[:,1], wps[:,2], marker='x', s=50, label='waypoints')
        ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_zlim(*zlim)
        try: ax.set_box_aspect((1,1,1))  # type: ignore
        except Exception: pass
        ax.set_xlabel('X [m]'); ax.set_ylabel('Y [m]'); ax.set_zlabel('Z [m]')
        ax.set_title('3D Trajectory (Fixed World Bounds)')
        ax.legend(loc='upper right', fontsize=8)
        ax.view_init(elev=elev, azim=azim)
        fig.tight_layout(); fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())

    try:
        iio.imwrite(out_mp4, frames, fps=fps)
        print(f"[SAVE] 3D 动画：{os.path.abspath(out_mp4)} (fps={fps}, frames={len(frames)})")
    except Exception as e:
        print(f"[WARN] MP4 写入失败：{e}")


# ==================================================
# 工具
# ==================================================

class DroneState:
    __slots__ = ("pos", "vel")
    def __init__(self, pos: np.ndarray, vel: np.ndarray):
        self.pos = pos.reshape(3); self.vel = vel.reshape(3)

def _get_state_from_obs(obs, idx: int = 0) -> 'DroneState':
    if isinstance(obs, dict):
        if "x" in obs and "v" in obs:
            pos, vel = np.array(obs["x"][idx], float), np.array(obs["v"][idx], float)
        elif "position" in obs and "velocity" in obs:
            pos, vel = np.array(obs["position"][idx], float), np.array(obs["velocity"][idx], float)
        else: raise KeyError("obs 中找不到位置/速度")
    elif isinstance(obs, np.ndarray):
        pos, vel = np.asarray(obs[idx,0:3], float), np.asarray(obs[idx,3:6], float)
    else:
        raise TypeError(f"未知 obs 类型: {type(obs)}")
    return DroneState(pos, vel)

def ensure_env_double(env) -> None:
    try:
        if hasattr(env, "quadrotors") and hasattr(env.quadrotors, "params"):
            params = env.quadrotors.params
            if hasattr(params, "inertia"): params.inertia = params.inertia.double()
        if hasattr(env, "vehicle_states") and isinstance(env.vehicle_states, dict):
            for k, v in env.vehicle_states.items():
                try: env.vehicle_states[k] = v.double()
                except Exception: pass
    except Exception as e:
        print("[WARN] ensure_env_double:", e)

def p_vel_to_waypoint(pos: np.ndarray, wp: np.ndarray, kp: float = C.KP, vmax: float = C.VMAX) -> np.ndarray:
    v = kp * (wp - pos).astype(np.float64)
    n = np.linalg.norm(v)
    if n > vmax > 0: v *= (vmax / n)
    return v

def speed_norm(vel: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(vel, dtype=np.float64)))

def _safe_quat_to_numpy(q):
    try:
        if torch is not None and isinstance(q, torch.Tensor):
            return q.detach().cpu().double().numpy().reshape(4)
    except Exception: pass
    return np.asarray(q, np.float64).reshape(4)

def _any_done(d): return (isinstance(d, bool) and d) or (isinstance(d, np.ndarray) and np.any(d))

def _clip_to_world(p: np.ndarray) -> np.ndarray:
    lim = max(C.WORLD_HALF_EXTENT - C.WORLD_MARGIN, 0.0)
    return np.clip(p, -lim, lim)

def _force_inject_state(env, init: Dict[str, np.ndarray]):
    try:
        vs = env.vehicle_states
        if not isinstance(vs, dict): return None
        mapping = {"x":["x","position"], "v":["v","velocity"], "q":["q"], "w":["w"],
                   "wind":["wind"], "rotor_speeds":["rotor_speeds"]}
        for src, keys in mapping.items():
            if src not in init: continue
            val = init[src]
            val_t = None
            try: val_t = torch.as_tensor(val, dtype=torch.float64) if torch is not None else None
            except Exception: pass
            for k in keys:
                if k in vs:
                    try: vs[k] = val_t.clone() if val_t is not None else np.array(val, np.float64, copy=True)
                    except Exception: vs[k] = val
        return {"x": np.asarray(init["x"], np.float64), "v": np.asarray(init["v"], np.float64)}
    except Exception as e:
        print("[WARN] _force_inject_state 失败：", e); return None

def _safe_reset_and_inject_with_probe(env, st_like: DroneState,
    wp_anchor_xy: Optional[np.ndarray], max_tries: int = 4, anchor_mode: str = "hold_xy",
    init_shrink: Optional[float] = None):
    """柔和重入：把无人机“放回”指定位置"""
    try: q_now = _safe_quat_to_numpy(env.vehicle_states["q"][0])
    except Exception: q_now = np.array([1.0,0.0,0.0,0.0], np.float64)

    if anchor_mode == "hold_xy":
        anchor, shrink = st_like.pos.copy(), 1.0
    elif anchor_mode == "toward_wp" and wp_anchor_xy is not None:
        anchor = np.array([wp_anchor_xy[0], wp_anchor_xy[1], st_like.pos[2]], np.float64)
        shrink = 0.9 if init_shrink is None else float(init_shrink)
    else:
        anchor, shrink = st_like.pos.copy(), 1.0

    obs = env.reset(); ensure_env_double(env); last_obs = obs
    for _ in range(max_tries):
        pos_try = _clip_to_world(anchor + shrink * (st_like.pos - anchor))
        init = {
            "x": pos_try.reshape(1,3),
            "v": st_like.vel.reshape(1,3),
            "q": np.tile(q_now, (1,1)),
            "w": np.zeros((1,3), np.float64),
            "wind": np.zeros((1,3), np.float64),
            "rotor_speeds": np.zeros((1,4), np.float64),
        }
        last_obs = _force_inject_state(env, init) or obs
        try: obs_test, _r, done, _i = env.step(np.zeros((1,3), np.float64))
        except Exception as e:
            print("[WARN] 探测 step 异常：", e); done, obs_test = True, last_obs
        ok = (isinstance(done, bool) and not done) or (isinstance(done, np.ndarray) and not np.any(done))
        if ok: return obs_test
        shrink *= 0.5
    print("[RE-ENTRY] 使用最后注入位置进入。"); return last_obs

def _done_reason(env, st_after_step: Optional[DroneState]) -> str:
    try:
        if hasattr(env, "t") and hasattr(env, "max_time"):
            t0 = float(np.array(env.t)[0]) if np.ndim(env.t) else float(env.t)
            if t0 >= float(env.max_time) - 1e-9: return "TIME_LIMIT"
    except Exception: pass
    try:
        st = st_after_step or _get_state_from_obs(env.vehicle_states, 0)
        if np.any(np.abs(st.pos) > (C.WORLD_HALF_EXTENT - 0.2)): return "WORLD_BOUNDARY"
    except Exception: pass
    return "UNKNOWN"


# ==================================================
# 控制器 & 航点
# ==================================================

def olfactory_nav_controller(conc: float, wind_vec_xy: Tuple[float, float],
                             pos: np.ndarray, z_ref: float, vmax: float = C.VMAX) -> np.ndarray:
    """嗅觉控制：>阈值逆风；<=阈值横风；极低浓度 BOOST"""
    Ux, Uy = float(wind_vec_xy[0]), float(wind_vec_xy[1])
    wind_dir = math.atan2(Uy, Ux)

    # 档位
    SPEED_UPWIND = 6.0   # 逆风追源
    SPEED_CROSS  = 4.0   # 横风搜索
    SPEED_BOOST  = 10.0  # 极低浓度外扩

    if conc > C.CONC_THRESHOLD:
        target_hdg = wind_dir + math.pi
        v_h_mag    = SPEED_UPWIND
    elif conc > 0.1 * C.CONC_THRESHOLD:
        target_hdg = wind_dir + math.pi/2.0
        v_h_mag    = SPEED_CROSS
    else:
        target_hdg = wind_dir + math.pi/2.0
        v_h_mag    = SPEED_BOOST

    vx_h, vy_h = math.cos(target_hdg), math.sin(target_hdg)
    vz = C.ALT_KP * (float(z_ref) - float(pos[2]))

    v_cmd = np.array([v_h_mag*vx_h, v_h_mag*vy_h, vz], dtype=np.float64)
    n = float(np.linalg.norm(v_cmd))
    if n > vmax > 0: v_cmd *= (vmax / n)
    return v_cmd


def gen_random_wp_local(st: 'DroneState', r_step: float = C.LOCAL_R_STEP) -> np.ndarray:
    ang = np.random.uniform(-np.pi, np.pi)
    r   = np.random.uniform(0.2*r_step, r_step)
    dx, dy = r*np.cos(ang), r*np.sin(ang)
    z  = np.random.uniform(C.Z_MIN, C.Z_MAX)
    p  = st.pos + np.array([dx, dy, z - st.pos[2]], dtype=np.float64)
    return _clip_to_world(p)


# ==================================================
# 主程序
# ==================================================

def main():
    print("[BOOT] 程序启动")
    print("[START] main_random_wps (olfactory OR random waypoints)")

    os.makedirs(C.OUTPUT_DIR, exist_ok=True)

    # 初始（env 会再 reset；我们随后强制注入实际起点）
    init = {
        "x": np.zeros((1,3), np.float64),
        "v": np.zeros((1,3), np.float64),
        "q": np.tile(np.array([1.0,0.0,0.0,0.0], np.float64), (1,1)),
        "w": np.zeros((1,3), np.float64),
        "wind": np.zeros((1,3), np.float64),
        "rotor_speeds": np.zeros((1,4), np.float64),
    }

    env = QuadrotorEnv(num_envs=1, initial_states=init, control_mode="cmd_vel", max_time=1e6)
    obs = env.reset(); ensure_env_double(env)

    # === PuffPlume ===
    plume = PuffPlume()

    # === 强制起点：远离源（确保真的从大域起飞）===
    try:
        src = np.array(plume.SRC, dtype=np.float64).reshape(3)
    except Exception:
        src = np.array([0.0,0.0,1.5], dtype=np.float64)
    desired_start = src + np.array([-200.0, -120.0, 0.0], dtype=np.float64)  # 左下角远处
    desired_start[2] = C.DESIRED_ALT
    st_like = DroneState(pos=desired_start, vel=np.zeros(3, np.float64))
    obs = _safe_reset_and_inject_with_probe(env, st_like, wp_anchor_xy=None,
                                            max_tries=4, anchor_mode="hold_xy",
                                            init_shrink=0.9)
    print(f"[INFO] 强制起点: {desired_start.tolist()}  源: {src.tolist()}")

    st0 = _get_state_from_obs(obs, 0)
    print(f"[INFO] 初始状态: pos={st0.pos}, vel={st0.vel}, |v|={speed_norm(st0.vel):.3f} m/s")

    # 日志
    rows: List[List[str]] = []
    xs: List[float] = []; ys: List[float] = []; zs: List[float] = []
    def log_state(t_s: float, phase: str, st: 'DroneState', wp=None, v_cmd=None, conc: float | None = None):
        rows.append([
            f"{t_s:.2f}", phase,
            f"{st.pos[0]:.6f}", f"{st.pos[1]:.6f}", f"{st.pos[2]:.6f}",
            f"{st.vel[0]:.6f}", f"{st.vel[1]:.6f}", f"{st.vel[2]:.6f}",
            f"{speed_norm(st.vel):.6f}",
            "" if wp is None else f"{wp[0]:.6f}",
            "" if wp is None else f"{wp[1]:.6f}",
            "" if wp is None else f"{wp[2]:.6f}",
            "" if v_cmd is None else f"{v_cmd[0]:.6f}",
            "" if v_cmd is None else f"{v_cmd[1]:.6f}",
            "" if v_cmd is None else f"{v_cmd[2]:.6f}",
            "" if conc is None else f"{conc:.8e}",
        ])

    DT = C.DT_GLOBAL
    t_s = 0.0
    rand_waypoints: List[np.ndarray] = []
    wp_count = 0

    if not C.USE_OLFACTORY_NAV:
        # -------- 随机航点（保留原逻辑）--------
        while True:
            if wp_count >= C.NUM_RANDOM_WP:
                print(f"[STOP] 已完成 {C.NUM_RANDOM_WP} 个随机航点，退出。")
                break
            st_now = _get_state_from_obs(obs,0)
            wp = gen_random_wp_local(st_now, r_step=C.LOCAL_R_STEP)
            rand_waypoints.append(wp.copy()); wp_count += 1
            print(f"[NAV] 追踪随机航点 {wp_count}/{C.NUM_RANDOM_WP}: {wp.tolist()}")

            dist0 = float(np.linalg.norm(wp - st_now.pos))
            t_allow = max(6.0, dist0 / (0.9 * C.VMAX) + 2.0)

            reached = False; t_wp_start = t_s; step = 0
            while True:
                st = _get_state_from_obs(obs, 0)
                xs.append(float(st.pos[0])); ys.append(float(st.pos[1])); zs.append(float(st.pos[2]))
                dist = float(np.linalg.norm(wp - st.pos))
                if dist <= 0.50:
                    plume.step(DT)
                    conc_reach = plume.sample(float(st.pos[0]), float(st.pos[1]), float(st.pos[2]))
                    log_state(t_s, f"REACH_WP_{wp_count}", st, wp=wp, v_cmd=np.array([0,0,0], np.float64), conc=conc_reach)
                    reached = True; break
                if (t_s - t_wp_start) >= t_allow:
                    plume.step(DT)
                    conc_timeout = plume.sample(float(st.pos[0]), float(st.pos[1]), float(st.pos[2]))
                    log_state(t_s, f"TIMEOUT_WP_{wp_count}", st, wp=wp, v_cmd=np.array([0,0,0], np.float64), conc=conc_timeout)
                    break

                v_cmd = p_vel_to_waypoint(st.pos, wp, kp=C.KP, vmax=C.VMAX)
                _n = float(np.linalg.norm(v_cmd))
                if (dist > C.KICK_DIST_FACTOR * 0.50) and (_n < C.VMIN_KICK) and (_n > 1e-9):
                    v_cmd = v_cmd * (C.VMIN_KICK / _n)

                ensure_env_double(env)
                obs, _r, done, _i = env.step(v_cmd.reshape(1,3))
                plume.step(DT)
                conc_now = plume.sample(float(st.pos[0]), float(st.pos[1]), float(st.pos[2]))
                log_state(t_s, f"TRACK_WP_{wp_count}", st, wp=wp, v_cmd=v_cmd, conc=conc_now)

                if step % 10 == 0:
                    print(f"[RUN] rand-wp#{wp_count} t={t_s:5.2f}s pos={st.pos} dist={dist:.2f} |v|={speed_norm(st.vel):.2f} conc={conc_now:.3e}")
                t_s += DT; step += 1
                if _any_done(done):
                    reason = _done_reason(env, _get_state_from_obs(obs, 0) if isinstance(obs,(dict,np.ndarray)) else None)
                    plume.step(DT)
                    conc_done = plume.sample(float(st.pos[0]), float(st.pos[1]), float(st.pos[2]))
                    print(f"[DONE] done={reason}，原地重入；跳过该航点。 pos={st.pos}")
                    obs = _safe_reset_and_inject_with_probe(env, st, wp_anchor_xy=None, max_tries=4, anchor_mode="hold_xy", init_shrink=0.9)
                    log_state(t_s, f"SKIP_WP_{wp_count}_DONE", st, wp=wp, v_cmd=np.array([0,0,0], np.float64), conc=conc_done)
                    break
                time.sleep(DT)

            if reached and C.HOVER_AFTER_REACH > 0.0:
                for _ in range(max(1, int(C.HOVER_AFTER_REACH/DT))):
                    st_h = _get_state_from_obs(obs, 0)
                    ensure_env_double(env)
                    obs, _r, done, _i = env.step(np.zeros((1,3), np.float64))
                    plume.step(DT); conc_hover = plume.sample(float(st_h.pos[0]), float(st_h.pos[1]), float(st_h.pos[2]))
                    log_state(t_s, f"HOVER_WP_{wp_count}", st_h, wp=wp, v_cmd=np.array([0,0,0], np.float64), conc=conc_hover)
                    xs.append(float(st_h.pos[0])); ys.append(float(st_h.pos[1])); zs.append(float(st_h.pos[2]))
                    t_s += DT; time.sleep(DT)

    else:
        # -------- 嗅觉导航（BOOST 版）--------
        print("[NAV-OLF] 嗅觉导航：阈值逆风；低浓度横风搜索；极低浓度 BOOST 外扩。")
        max_steps = int(1200.0 / DT)  # 最多 1200s 防止无限循环
        SRC = np.array(plume.SRC if hasattr(plume, "SRC") else [0.0,0.0,1.5], dtype=np.float64).reshape(3)
        MIN_STEPS_BEFORE_CHECK = int(C.MIN_OLF_TIME_SEC / DT)

        for step in range(max_steps):
            st = _get_state_from_obs(obs, 0)
            xs.append(float(st.pos[0])); ys.append(float(st.pos[1])); zs.append(float(st.pos[2]))

            plume.step(DT)
            conc_now = plume.sample(float(st.pos[0]), float(st.pos[1]), float(st.pos[2]))
            Ux, Uy = plume.wind_fn(plume.t)

            v_cmd = olfactory_nav_controller(conc_now, (Ux,Uy), st.pos, z_ref=C.DESIRED_ALT, vmax=C.VMAX)

            ensure_env_double(env)
            obs, _r, done, _i = env.step(v_cmd.reshape(1,3))

            log_state(t_s, "OLF_TRACK", st, wp=None, v_cmd=v_cmd, conc=conc_now)

            if step % 50 == 0:
                wind_dir_deg = math.degrees(math.atan2(Uy, Ux))
                print(f"[OLF] t={t_s:7.2f}s pos={st.pos} conc={conc_now:.3e} wind=({Ux:.2f},{Uy:.2f}) dir≈{wind_dir_deg:6.1f}° v_cmd={v_cmd}")

            # 到达联合判定（仅在最小飞行时长后检查）
            if step >= MIN_STEPS_BEFORE_CHECK:
                xy_ok = np.linalg.norm(st.pos[:2] - SRC[:2]) <= C.XY_REACH_R
                z_ok  = abs(st.pos[2] - SRC[2]) <= C.Z_REACH_R
                conc_ok = (conc_now > C.CONC_THRESHOLD*0.2) if C.REACH_NEED_CONC else True
                if xy_ok and z_ok and conc_ok:
                    print(f"[OLF] Reached source region: pos={st.pos}, conc={conc_now:.3e}, xy<= {C.XY_REACH_R}m, |dz|<= {C.Z_REACH_R}m")
                    break

            t_s += DT
            if _any_done(done):
                reason = _done_reason(env, _get_state_from_obs(obs, 0) if isinstance(obs,(dict,np.ndarray)) else None)
                print(f"[OLF-DONE] env done={reason}，原地重入继续。")
                obs = _safe_reset_and_inject_with_probe(env, st, wp_anchor_xy=None, max_tries=4, anchor_mode="hold_xy", init_shrink=0.9)

            time.sleep(DT)

        print("[NAV-OLF] 嗅觉导航结束。")

    # ===== 输出 =====
    xs_a, ys_a, zs_a = np.asarray(xs,float), np.asarray(ys,float), np.asarray(zs,float)
    save_csv(rows, C.CSV_FILE)
    try:
        plot_static(xs_a, ys_a, zs_a, rand_waypoints, dt=C.DT_GLOBAL,
                    fig_traj_3d=C.FIG_TRAJ_3D, fig_traj_xy=C.FIG_TRAJ_XY, fig_alt_time=C.FIG_ALT_TIME)
    except Exception as e:
        print("[WARN] 绘图失败：", e)
    render_3d_animation(xs_a, ys_a, zs_a, rand_waypoints,
                        out_mp4=C.ANIM_3D_MP4, fps=C.ANIM_FPS, stride=C.ANIM_STRIDE,
                        elev=C.ANIM_ELEV, azim=C.ANIM_AZIM)

    st_end = _get_state_from_obs(obs, 0)
    print(f"[EXIT] 结束: pos={st_end.pos}, vel={st_end.vel}, |v|={speed_norm(st_end.vel):.3f} m/s")


if __name__ == "__main__":
    try:
        main()
    finally:
        print("[SHUTDOWN] 程序退出")
