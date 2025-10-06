
"""
- 使用 RotorPy 的 QuadrotorEnv 中按航点序列导航；
- 采用简单的 P 控制器在速度空间追踪航点，并限制最大速度；
- 为每个航点设置“单航点时间阈值”（超时则跳过该航点）；
- 仅在 env 报 done=True（例如越界/内部时限等）时，执行“重入（reset+注入+探测）”，可选择朝航点方向轻微收缩或原地重入；
- 输出 CSV、静态图（3D、XY、高度-时间）以及3D动画；
"""

import os, csv, time
from typing import Dict, Optional, List
import numpy as np

try:
    import torch
    torch.set_default_dtype(torch.float64)  # 统一为 float64 精度
except Exception:
    torch = None

# RotorPy 环境
from rotorpy.learning.quadrotor_environments import QuadrotorEnv


# =========================
# 一、全局参数
# =========================

DT_GLOBAL              = 0.02      # 控制回路周期（s），50Hz
KP                     = 1.2       # P控制器（速度指令 = KP * 位置误差）
VMAX                   = 0.6       # 速度上限（m/s），用于限幅速度指令
WP_TOL                 = 0.10      # 判定到达航点的距离阈值（m）
MAX_TIME_PER_WP        = 6.0       # “单航点时间阈值”（秒）：超过则跳过该航点
MAX_REENTRY_PER_WP     = 2         # 同一航点内最多允许 done 后重入次数 （33-38： 防止超出边界或其他错误）
MIN_RUN_AFTER_REENTRY  = 0.5       # 重入后至少运行这么久，才计数下一次重入（秒）
REENTRY_INIT_SHRINK    = 0.7       # 重入注入时的初始收缩系数（朝锚点/航点方向压缩 30%）
REENTRY_SHRINK_FACTOR  = 0.5       # 若探测仍 done，则继续缩小的倍率
WORLD_HALF_EXTENT      = 4.0       # 世界边界半边长（±4m），与 RotorPy 默认一致
WORLD_MARGIN           = 0.30      # 注入时留一层安全边距，避免紧贴边界
HOVER_AFTER_REACH      = 0.2       # 到达航点后的“短悬停时间”（秒）。设为 0 可关闭

# 航点：设定在 ±3m以内（边界值）
WAYPOINTS: List[np.ndarray] = [
    np.array([ 0.8, -0.8, 0.9 ], dtype=np.float64),
    np.array([-1.5,  1.2, 1.4 ], dtype=np.float64),
    np.array([ 2.2,  0.5, 1.0 ], dtype=np.float64),
    np.array([ 0.0,  2.4, 1.6 ], dtype=np.float64),
    np.array([-2.8, -1.8, 1.2 ], dtype=np.float64),
    np.array([ 0.6,  0.0, 0.8 ], dtype=np.float64),
]


# =========================
# 二、工具
# =========================

class DroneState:
    """轻量状态封装：仅包含位置与速度向量（均为 np.ndarray shape=(3,)）。"""
    __slots__ = ("pos", "vel")
    def __init__(self, pos: np.ndarray, vel: np.ndarray):
        self.pos = pos
        self.vel = vel


def _get_state_from_obs(obs, idx: int = 0) -> DroneState:
    """
    将 RotorPy 的 obs 解析为 DroneState。
    - dict 可能有 (x,v) 或 (position, velocity) 这两套命名；
    - ndarray 视为 [x,y,z,vx,vy,vz,...] 的扁平化排布（取前 6 项）。
    """
    if isinstance(obs, dict):
        if "x" in obs and "v" in obs:
            pos, vel = np.array(obs["x"][idx], float), np.array(obs["v"][idx], float)
        elif "position" in obs and "velocity" in obs:
            pos, vel = np.array(obs["position"][idx], float), np.array(obs["velocity"][idx], float)
        else:
            raise KeyError("obs 中找不到位置/速度（x,v 或 position,velocity）")
    elif isinstance(obs, np.ndarray):
        pos, vel = np.asarray(obs[idx, 0:3], float), np.asarray(obs[idx, 3:6], float)
    else:
        raise TypeError(f"未知 obs 类型: {type(obs)}")
    return DroneState(pos.reshape(3), vel.reshape(3))


def ensure_env_double(env) -> None:
    """
    将环境内部的张量（尤其是惯量矩阵 inertia）切换为 float64，
    避免与我们 np.float64 指令混用时报错（Float vs Double）。
    """
    try:
        if hasattr(env, "quadrotors") and hasattr(env.quadrotors, "params"):
            params = env.quadrotors.params
            if hasattr(params, "inertia"):
                params.inertia = params.inertia.double()
        if hasattr(env, "vehicle_states") and isinstance(env.vehicle_states, dict):
            for k, v in env.vehicle_states.items():
                try:
                    env.vehicle_states[k] = v.double()
                except Exception:
                    pass
    except Exception as e:
        print("[WARN] ensure_env_double:", e)


# =========================
# 三、控制器：P 控制
# =========================

def p_vel_to_waypoint(pos: np.ndarray, wp: np.ndarray, kp: float = KP, vmax: float = VMAX) -> np.ndarray:
    """
    简单 P 控制器：根据位置误差生成速度指令，并进行幅值限幅。
    v_cmd = clip( KP * (wp - pos), |v| <= VMAX )
    """
    v = kp * (wp - pos).astype(np.float64)
    n = np.linalg.norm(v)
    if n > vmax > 0:
        v *= (vmax / n)
    return v


def speed_norm(vel: np.ndarray) -> float:
    """返回整体速度标量 |v|。"""
    return float(np.linalg.norm(np.asarray(vel, dtype=np.float64)))


# =========================
# 四. done 时使用）
# =========================

def _safe_quat_to_numpy(q):
    """将四元数安全转为 np.float64[4]。支持 torch.Tensor 或 numpy。"""
    try:
        if torch is not None and isinstance(q, torch.Tensor):
            return q.detach().cpu().double().numpy().reshape(4)
    except Exception:
        pass
    return np.asarray(q, np.float64).reshape(4)


def _any_done(d):
    """done 兼容：bool 或 np.ndarray（batched）。"""
    return (isinstance(d, bool) and d) or (isinstance(d, np.ndarray) and np.any(d))


def _clip_to_world(p: np.ndarray, half_extent: float = WORLD_HALF_EXTENT, margin: float = WORLD_MARGIN) -> np.ndarray:
    """将注入位置裁剪到 ±(half_extent - margin) 的立方体内。"""
    lim = max(half_extent - margin, 0.0)
    return np.clip(p, -lim, lim)


def _force_inject_state(env, init: Dict[str, np.ndarray]):

    try:
        vs = env.vehicle_states
        if not isinstance(vs, dict):
            return None
        mapping = {"x": ["x","position"], "v": ["v","velocity"], "q": ["q"], "w": ["w"],
                   "wind": ["wind"], "rotor_speeds": ["rotor_speeds"]}
        for src, keys in mapping.items():
            if src not in init: 
                continue
            val = init[src]
            val_t = None
            try:
                val_t = torch.as_tensor(val, dtype=torch.float64) if torch is not None else None
            except Exception:
                pass
            for k in keys:
                if k in vs:
                    try:
                        vs[k] = val_t.clone() if val_t is not None else np.array(val, np.float64, copy=True)
                    except Exception:
                        vs[k] = val
        return {"x": np.asarray(init["x"], np.float64), "v": np.asarray(init["v"], np.float64)}
    except Exception as e:
        print("[WARN] _force_inject_state 失败：", e)
        return None


def _safe_reset_and_inject_with_probe(env, st_like: DroneState, wp_anchor_xy: Optional[np.ndarray],
                                      max_tries: int = 8, anchor_mode: str = "toward_wp", init_shrink: Optional[float] = None):
    """
    仅在 env 报 done=True 时调用：
      reset → 注入 → 发 0 动作探测 → 若仍 done 则缩小并重试，直到可运行。
    anchor_mode:
      - "toward_wp": 朝“当前航点的 (x,y) + 当前 z”的锚点方向进行收缩注入。
      - "hold_xy"  : 原地重入。
    """
    # 取当前姿态 q
    try:
        q_now = _safe_quat_to_numpy(env.vehicle_states["q"][0])
    except Exception:
        q_now = np.array([1.0, 0.0, 0.0, 0.0], np.float64)

    # 计算锚点与初始 shrink
    if anchor_mode == "toward_wp" and wp_anchor_xy is not None:
        anchor = np.array([wp_anchor_xy[0], wp_anchor_xy[1], st_like.pos[2]], np.float64)
        shrink = REENTRY_INIT_SHRINK if init_shrink is None else float(init_shrink)
    elif anchor_mode == "hold_xy":
        anchor, shrink = st_like.pos.copy(), 1.0
    else:
        anchor, shrink = np.array([0.0, 0.0, st_like.pos[2]], np.float64), (REENTRY_INIT_SHRINK if init_shrink is None else float(init_shrink))

    # reset
    obs = env.reset()
    ensure_env_double(env)
    last_obs = obs

    for k in range(max_tries):
        # （裁剪以避免贴边）
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

        # 若仍 done，缩小收缩比并重试
        try:
            obs_test, _r, done, _i = env.step(np.zeros((1,3), np.float64))
        except Exception as e:
            print("[WARN] 探测 step 异常：", e)
            done, obs_test = True, last_obs

        ok = (isinstance(done, bool) and not done) or (isinstance(done, np.ndarray) and not np.any(done))
        if ok:
            if k > 0:
                print(f"[RE-ENTRY] 探测成功（第{k+1}次），shrink={shrink:.3f}")
            return obs_test

        shrink *= float(REENTRY_SHRINK_FACTOR)

    print("[RE-ENTRY] 尝试用尽，使用最后注入位置进入。")
    return last_obs


def _done_reason(env, st_after_step: Optional[DroneState]) -> str:
    """
     done 原因：
    - TIME_LIMIT     : env.t 达到 env.max_time（RotorPy 的“单次会话上限”）
    - WORLD_BOUNDARY : 位置触及近边界（粗判，> WORLD_HALF_EXTENT - 0.2）
    - UNKNOWN        : 其它原因（比如数值异常等）
    """
    try:
        if hasattr(env, "t") and hasattr(env, "max_time"):
            t0 = float(np.array(env.t)[0]) if np.ndim(env.t) else float(env.t)
            if t0 >= float(env.max_time) - 1e-9:
                return "TIME_LIMIT"
    except Exception:
        pass
    try:
        st = st_after_step or _get_state_from_obs(env.vehicle_states, 0)
        if np.any(np.abs(st.pos) > (WORLD_HALF_EXTENT - 0.2)):
            return "WORLD_BOUNDARY"
    except Exception:
        pass
    return "UNKNOWN"


# =========================
# 五. plot, 3D动画
# =========================

def plot_static(xs: np.ndarray, ys: np.ndarray, zs: np.ndarray, waypoints: List[np.ndarray], dt: float):
    """保存 3 张静态图：3D 轨迹、XY 俯视、高度-时间"""
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    except Exception as e:
        print("[WARN] 绘图跳过:", e); return

    # 3D 轨迹
    fig = plt.figure(figsize=(6.08, 6.08), dpi=100)
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(xs, ys, zs, label='trajectory')
    ax.scatter(xs[0], ys[0], zs[0], marker='o', s=40, label='start')
    ax.scatter(xs[-1], ys[-1], zs[-1], marker='^', s=40, label='end')
    if waypoints:
        wps = np.stack(waypoints, 0)
        ax.scatter(wps[:,0], wps[:,1], wps[:,2], marker='x', s=50, label='waypoints')
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z'); ax.set_title('3D Trajectory')
    try: ax.set_box_aspect((1,1,1))
    except Exception: pass
    ax.legend(); fig.tight_layout(); fig.savefig('trajectory_3d.png', dpi=150); plt.close(fig)
    print("[SAVE] 3D 轨迹图：trajectory_3d.png")

    # XY 俯视
    fig = plt.figure(figsize=(6.08, 6.08), dpi=100)
    ax = fig.add_subplot(111)
    ax.plot(xs, ys, label='trajectory')
    ax.scatter(xs[0], ys[0], marker='o', s=40, label='start')
    ax.scatter(xs[-1], ys[-1], marker='^', s=40, label='end')
    if waypoints:
        ax.scatter(wps[:,0], wps[:,1], s=50, marker='x', label='waypoints')
        ax.plot(wps[:,0], wps[:,1], linestyle='--', label='wp path')
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_title('XY Top-Down')
    ax.axis('equal'); ax.grid(True); ax.legend()
    fig.tight_layout(); fig.savefig('trajectory_xy.png', dpi=150); plt.close(fig)
    print("[SAVE] XY 俯视图：trajectory_xy.png")

    # 高度-时间
    t = np.arange(len(zs))*dt
    fig = plt.figure(figsize=(7,3.5)); ax = fig.add_subplot(111)
    ax.plot(t, zs, label='z(t)'); ax.set_xlabel('Time [s]'); ax.set_ylabel('Z [m]'); ax.grid(True)
    ax.set_title('Altitude over Time')
    fig.tight_layout(); fig.savefig('altitude_time.png', dpi=150); plt.close(fig)
    print("[SAVE] 高度-时间：altitude_time.png")


def render_3d_animation(xs: np.ndarray, ys: np.ndarray, zs: np.ndarray, waypoints: List[np.ndarray],
                        out_mp4: str = "trajectory_3d.mp4", fps: int = 30, stride: int = 2,
                        elev: float = 25.0, azim: float = -60.0):
    """
    3D轨迹动画：
    """
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        import imageio.v3 as iio
    except Exception as e:
        print("[WARN] 3D 动画跳过（缺少依赖）：", e); return

    # 保持立方体比例的显示范围
    x_min, x_max = float(np.min(xs)), float(np.max(xs))
    y_min, y_max = float(np.min(ys)), float(np.max(ys))
    z_min, z_max = float(np.min(zs)), float(np.max(zs))
    span = max(x_max - x_min, y_max - y_min, z_max - z_min, 1e-6)
    cx, cy, cz = (0.5*(x_min+x_max), 0.5*(y_min+y_max), 0.5*(z_min+z_max))
    pad = 0.10 * span
    xlim = (cx - span/2 - pad, cx + span/2 + pad)
    ylim = (cy - span/2 - pad, cy + span/2 + pad)
    zlim = (cz - span/2 - pad, cz + span/2 + pad)

    fig = plt.figure(figsize=(6.08, 6.08), dpi=100)
    ax = fig.add_subplot(111, projection='3d')
    wps = np.stack(waypoints, 0) if waypoints else None

    frames: List[np.ndarray] = []
    idxs = list(range(1, len(xs), stride))
    if idxs and idxs[-1] != len(xs) - 1:
        idxs.append(len(xs) - 1)

    for idx in idxs:
        ax.clear()
        ax.plot(xs[:idx], ys[:idx], zs[:idx], label='trajectory')
        ax.scatter(xs[0], ys[0], zs[0], marker='o', s=40, label='start')
        ax.scatter(xs[idx-1], ys[idx-1], zs[idx-1], marker='^', s=40, label='current')
        if wps is not None:
            ax.scatter(wps[:, 0], wps[:, 1], wps[:, 2], marker='x', s=50, label='waypoints')
            ax.plot(wps[:, 0], wps[:, 1], wps[:, 2], linestyle='--', label='wp path')
        ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_zlim(*zlim)
        try: ax.set_box_aspect((1, 1, 1))
        except Exception: pass
        ax.set_xlabel('X [m]'); ax.set_ylabel('Y [m]'); ax.set_zlabel('Z [m]')
        ax.set_title('3D Trajectory (Fixed View)')
        ax.legend(loc='upper right', fontsize=8)
        ax.view_init(elev=elev, azim=azim)
        fig.tight_layout(); fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())

    try:
        iio.imwrite(out_mp4, frames, fps=fps)
        print(f"[SAVE] 3D 动画：{os.path.abspath(out_mp4)}  (fps={fps}, frames={len(frames)})")
    except Exception as e:
        print(f"[WARN] MP4 写入失败：{e}")


# =========================
# 六、main
# =========================

def main():
    print("[START] drone_track_slim_v_with_docs")

    # 1) 初始状态（全部为 0，姿态为单位四元数）
    init = {
        "x": np.zeros((1,3), np.float64),
        "v": np.zeros((1,3), np.float64),
        "q": np.tile(np.array([1.0,0.0,0.0,0.0], np.float64), (1,1)),
        "w": np.zeros((1,3), np.float64),
        "wind": np.zeros((1,3), np.float64),
        "rotor_speeds": np.zeros((1,4), np.float64),
    }

    # 2) 创建环境
    env = QuadrotorEnv(num_envs=1, initial_states=init, control_mode="cmd_vel", max_time=1e6)

    # 3) 并确保内部dtype
    obs = env.reset()
    ensure_env_double(env)

    #初始状态
    st0 = _get_state_from_obs(obs, 0)
    print(f"[INFO] 初始状态: pos={st0.pos}, vel={st0.vel}, |v|={speed_norm(st0.vel):.3f} m/s")

    # 4) CSV 行、轨迹数组
    rows: List[List] = []  # type: ignore
    xs: List[float] = []
    ys: List[float] = []
    zs: List[float] = []

    # 记录：追加一行到 CSV 缓存
    def log_state(t_s: float, phase: str, st: DroneState, wp=None, v_cmd=None):
        rows.append([
            f"{t_s:.2f}", phase,
            f"{st.pos[0]:.6f}", f"{st.pos[1]:.6f}", f"{st.pos[2]:.6f}",
            f"{st.vel[0]:.6f}", f"{st.vel[1]:.6f}", f"{st.vel[2]:.6f}",
            f"{speed_norm(st.vel):.6f}",  # 整体速度 |v|
            "" if wp is None else f"{wp[0]:.6f}",
            "" if wp is None else f"{wp[1]:.6f}",
            "" if wp is None else f"{wp[2]:.6f}",
            "" if v_cmd is None else f"{v_cmd[0]:.6f}",
            "" if v_cmd is None else f"{v_cmd[1]:.6f}",
            "" if v_cmd is None else f"{v_cmd[2]:.6f}",
        ])

    # 统一控制周期
    DT = DT_GLOBAL
    t_s = 0.0  

    # 5) 航点 main
    for wi, wp in enumerate(WAYPOINTS):
        print(f"[NAV] 追踪航点 {wi+1}/{len(WAYPOINTS)}: {wp.tolist()}")
        reached = False
        reentry_count = 0
        t_wp_start = t_s
        t_last_reentry: Optional[float] = None

        step = 0
        while True:
            # a) 读取当前状态并入轨迹数组
            st = _get_state_from_obs(obs, 0)
            xs.append(float(st.pos[0])); ys.append(float(st.pos[1])); zs.append(float(st.pos[2]))

            # b) 到达判定
            dist = float(np.linalg.norm(wp - st.pos))
            if dist <= WP_TOL:
                print(f"[OK] 已到达航点#{wi+1}  dist={dist:.3f}  pos={st.pos}")
                log_state(t_s, f"REACH_WP_{wi+1}", st, wp=wp, v_cmd=np.array([0,0,0], np.float64))
                reached = True
                break

            # c) 超时判定（单航点时间阈值）
            elapsed = t_s - t_wp_start
            if elapsed >= MAX_TIME_PER_WP:
                print(f"[SKIP] 航点#{wi+1} 超时未到达（{elapsed:.2f}s ≥ {MAX_TIME_PER_WP:.2f}s），跳过。")
                log_state(t_s, f"TIMEOUT_WP_{wi+1}", st, wp=wp, v_cmd=np.array([0,0,0], np.float64))
                break  #保持当前位置直接进入下一个航点

            # d) 计算速度指令并推进一步
            v_cmd = p_vel_to_waypoint(st.pos, wp)
            ensure_env_double(env)
            obs, _r, done, _i = env.step(v_cmd.reshape(1,3))

            # 记录
            log_state(t_s, f"TRACK_WP_{wi+1}", st, wp=wp, v_cmd=v_cmd)

            # 每 0.4s 打印一次
            if step % 20 == 0:
                print(f"[RUN] wp#{wi+1} t={t_s:5.2f}s  pos={st.pos}  dist={dist:.3f}  v_cmd={v_cmd}  |v|={speed_norm(st.vel):.3f}  elapsed={elapsed:.2f}s")

            # e) 步进
            t_s += DT
            step += 1

            # f) done 处理
            if _any_done(done):
                reason = _done_reason(env, _get_state_from_obs(obs, 0) if isinstance(obs,(dict,np.ndarray)) else None)
                short = (t_last_reentry is not None) and ((t_s - t_last_reentry) < MIN_RUN_AFTER_REENTRY)
                if not short:
                    reentry_count += 1
                print(f"[DONE] WP#{wi+1} 触发 done（reason={reason}），重入第 {reentry_count} 次（保护窗内: {short}）。 pos={st.pos}")

                # 朝“当前航点方向”轻微收缩
                obs = _safe_reset_and_inject_with_probe(
                    env, st, wp_anchor_xy=wp, max_tries=8, anchor_mode="toward_wp", init_shrink=0.9
                )
                t_last_reentry = t_s

                if (reentry_count > MAX_REENTRY_PER_WP) and not short:
                    print(f"[SKIP] WP#{wi+1} 重入次数超限（{reentry_count}>{MAX_REENTRY_PER_WP}），跳过。")
                    log_state(t_s, f"SKIP_WP_{wi+1}", st, wp=wp, v_cmd=np.array([0,0,0], np.float64))
                    break

            # 
            time.sleep(DT)

        # h) 到达后短悬停
        if reached and HOVER_AFTER_REACH > 0.0:
            for _ in range(max(1, int(HOVER_AFTER_REACH / DT_GLOBAL))):
                st_h = _get_state_from_obs(obs, 0)
                ensure_env_double(env)
                obs, _r, done, _i = env.step(np.zeros((1,3), np.float64))
                if _any_done(done):
                    log_state(t_s, f"HOVER_WP_{wi+1}_DONE", st_h, wp=wp, v_cmd=np.array([0,0,0], np.float64))
                    print(f"[DONE] 悬停阶段触发 done（reason={_done_reason(env, None)}），原地重入继续。 pos={st_h.pos}")
                    obs = _safe_reset_and_inject_with_probe(env, st_h, wp_anchor_xy=None, max_tries=8, anchor_mode="hold_xy")
                    break
                log_state(t_s, f"HOVER_WP_{wi+1}", st_h, wp=wp, v_cmd=np.array([0,0,0], np.float64))
                xs.append(float(st_h.pos[0])); ys.append(float(st_h.pos[1])); zs.append(float(st_h.pos[2]))
                t_s += DT
                time.sleep(DT)

    # 6) 保存CSV
    with open("flight_log.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "time_s","phase",
            "x","y","z","vx","vy","vz","speed",  
            "wp_x","wp_y","wp_z","v_cmd_x","v_cmd_y","v_cmd_z"
        ])
        w.writerows(rows)
    print("[SAVE] 轨迹已保存：", os.path.abspath("flight_log.csv"))

    # 7) 保存plt,3D动画
    xs_a, ys_a, zs_a = np.asarray(xs,float), np.asarray(ys,float), np.asarray(zs,float)
    try:
        plot_static(xs_a, ys_a, zs_a, WAYPOINTS, dt=DT_GLOBAL)
    except Exception as e:
        print("[WARN] 绘图失败：", e)
    render_3d_animation(xs_a, ys_a, zs_a, WAYPOINTS, out_mp4="trajectory_3d.mp4", fps=30, stride=2)

    # 8) 结束状态
    st_end = _get_state_from_obs(obs, 0)
    print(f"[EXIT] 结束: pos={st_end.pos}, vel={st_end.vel}, |v|={speed_norm(st_end.vel):.3f} m/s")


if __name__ == "__main__":
    print("[BOOT] 程序启动")
    try:
        main()
    finally:
        print("[SHUTDOWN] 程序退出")
