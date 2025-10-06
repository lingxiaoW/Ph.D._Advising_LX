"""
--------------------------------------------------
- 创建 RotorPy 的四旋翼仿真环境（QuadrotorEnv，control_mode='cmd_vel'）。
- 无限循环 while True 中，依次生成“随机航点”（共 NUM_RANDOM_WP 个），
  使用简单的比例控制器（P 控制）产生速度指令 v_cmd，使无人机朝当前航点运动。
- 对于每个航点：
  * 若在距离阈值 WP_TOL 内到达 -> 记录 REACH 并进入短暂悬停阶段；
  * 若超过 MAX_TIME_PER_WP 未到达 -> 记录 TIMEOUT 并跳过此航点；
  * 若仿真返回 done（撞边界/到时） -> 执行“重入”（reset+状态注入+探测 step），
    以当前状态为参考把无人机“放回场景”，尽量继续当前航点；重入次数超过上限则跳过此航点。
- 全程按固定步长 DT_GLOBAL 推进时间，并把每一步的状态/指令写入内存列表 rows，
  最后统一调用 viz_and_io.py 输出 CSV 与图像/动画。

文件间职责
--------------------------------------------------
- config_drone.py：可调整参数（时间、速度、阈值、随机范围、输出文件名等）。
- viz_and_io.py：绘图与保存（CSV、平面图、3D 图、3D 动画）。
- main_random_wps.py（本文件）：仅保留“控制逻辑 + 重入 + 日志采集 + 调用输出”。

数据流/关键变量
--------------------------------------------------
- obs：环境返回的观测（dict 或 ndarray）。我们只用到“位置 x/position 与速度 v/velocity”。
- DroneState：便捷结构，封装 pos/vel（均为 shape=(3,) 的 numpy 向量）。
- t_s：仿真时间（秒），每一步累加 DT_GLOBAL。用于日志与时间阈值判断。
- rows：逐步累积的日志行（字符串），最后交给 save_csv。
- rand_waypoints：用于绘图展示的航点列表（np.array，形如 [x,y,z]）。

注意事项
--------------------------------------------------
- “done”产生的原因比较多。此处通过 _done_reason 进行粗略判断（时间上限、世界边界）。
- “重入”是一个工程性手段：试图用 env.reset()+状态注入+探测 step 找到“稳定的”起点继续仿真。
- 本逻辑为教学/实验性质，易读性优先；如果需要更高的物理/控制精度，建议替换为姿态/速度/位置闭环控制器。
- 本文件默认中文注释，便于逐行理解。
"""

import time
from typing import Dict, Optional, List, Tuple
import numpy as np

try:
    import torch
    torch.set_default_dtype(torch.float64)
except Exception:
    torch = None  

# 2) RotorPy 环境
from rotorpy.learning.quadrotor_environments import QuadrotorEnv
import config_drone as C
from viz_and_io import plot_static, render_3d_animation, save_csv


# ==================================================
# 基础结构体与通用工具函数
# ==================================================

class DroneState:
    """
    轻量级状态结构：仅包含位置 pos 与速度 vel。
    - pos: np.ndarray, shape=(3,), 单位 m
    - vel: np.ndarray, shape=(3,), 单位 m/s
    """
    __slots__ = ("pos", "vel")
    def __init__(self, pos: np.ndarray, vel: np.ndarray):
        self.pos = pos.reshape(3)
        self.vel = vel.reshape(3)


def _get_state_from_obs(obs, idx: int = 0) -> DroneState:
    """
    从环境观测 obs 中提取 DroneState。
    兼容两种常见结构：
      - dict: obs["x"][i], obs["v"][i] 或 obs["position"][i], obs["velocity"][i]
      - ndarray: obs[i, :6] -> [x,y,z,vx,vy,vz]
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
    return DroneState(pos, vel)


def ensure_env_double(env) -> None:
    """
    若环境内部使用了 torch.Tensor，这里尽量把 inertia / vehicle_states 等切到 double，
    避免 float32 带来的小误差放大（该函数是“尽力而为”，失败不会中断）。
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


def p_vel_to_waypoint(pos: np.ndarray, wp: np.ndarray, kp: float = C.KP, vmax: float = C.VMAX) -> np.ndarray:
    """
    位置 -> 速度（P 控制）：v_cmd = kp * (wp - pos)，并在范数上限处做饱和（<= VMAX）。
    - pos: 当前无人机位置
    - wp : 当前目标航点
    - kp : P 控制增益（影响“起步/响应”快慢）
    - vmax: 指令速度上限（影响巡航上限与快速到达能力）
    """
    v = kp * (wp - pos).astype(np.float64)
    n = np.linalg.norm(v)
    if n > vmax > 0:
        v *= (vmax / n)
    return v


def speed_norm(vel: np.ndarray) -> float:
    """返回速度模长 |v|，仅用于日志打印。"""
    return float(np.linalg.norm(np.asarray(vel, dtype=np.float64)))


def _safe_quat_to_numpy(q):
    """将四元数张量/数组安全转为 np.ndarray(float64)。用于重入时保留当前朝向。"""
    try:
        if torch is not None and isinstance(q, torch.Tensor):
            return q.detach().cpu().double().numpy().reshape(4)
    except Exception:
        pass
    return np.asarray(q, np.float64).reshape(4)


def _any_done(d):
    """
    统一判断环境的 done（可能是 bool，也可能是 numpy 向量）。
    - True 表示此步需要重置/重入（可能撞边界、时间耗尽等）。
    """
    return (isinstance(d, bool) and d) or (isinstance(d, np.ndarray) and np.any(d))


def _clip_to_world(p: np.ndarray) -> np.ndarray:
    """
    把点 p 裁剪到世界边界内（留下 WORLD_MARGIN 的安全边）。
    - 用于重入时，避免把无人机“放回”到边界之外。
    """
    lim = max(C.WORLD_HALF_EXTENT - C.WORLD_MARGIN, 0.0)
    return np.clip(p, -lim, lim)


def _force_inject_state(env, init: Dict[str, np.ndarray]):
    """
    强行把“我们想要的状态”注入到环境的 vehicle_states 里（工程 workaround）：
    - 在 env.reset() 之后调用：
        x / position        <- 初始位置 (1,3)
        v / velocity        <- 初始速度 (1,3)
        q                   <- 初始姿态四元数 (1,4)
        w, wind, rotor_speeds <- 清零
    - 返回值：一个“类似 obs”的字典（含 x/v），用于后续继续 step。
    - 若失败，打印告警但不中断。
    """
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


def _safe_reset_and_inject_with_probe(
    env,
    st_like: DroneState,
    wp_anchor_xy: Optional[np.ndarray],
    max_tries: int = 8,
    anchor_mode: str = "toward_wp",
    init_shrink: Optional[float] = None
):
    """
    “重入”策略（核心思想：reset -> 注入 -> 探测 step）
    --------------------------------------------------
    目标：在 done 之后，让无人机尽量“接近原来状态”或“朝向航点的合理位置”继续飞，
          而不是重新从环境默认原点起飞，避免实验被意外打断。

    参数说明：
    - st_like:      参考状态（通常取 done 前的最后状态）
    - wp_anchor_xy: 若 anchor_mode='toward_wp'，我们用这个航点的 XY 来作为锚点，
                    把 st_like 在 XY 上“拉”向航点，以期减少下一步的距离；
                    若为 None 或 anchor_mode='hold_xy'，则维持原 XY。
    - max_tries:    注入后的“探测 step”尝试次数。若 done 仍触发，则缩小位移（shrink）再试。
    - anchor_mode:  'toward_wp'（朝向航点锚定），或 'hold_xy'（保持当前位置为锚点）。
    - init_shrink:  初始缩放（默认 0.9）。每次失败后乘以 REENTRY_SHRINK_FACTOR 继续尝试。

    返回值：成功后的“可用观测”（obs-like），或最后一次注入的观测。
    """
    try:
        q_now = _safe_quat_to_numpy(env.vehicle_states["q"][0])
    except Exception:
        q_now = np.array([1.0, 0.0, 0.0, 0.0], np.float64)

    if anchor_mode == "toward_wp" and wp_anchor_xy is not None:
        # 以“航点的 XY + 原高度 Z”为锚点，把 st_like 向此锚点缩放移动
        anchor = np.array([wp_anchor_xy[0], wp_anchor_xy[1], st_like.pos[2]], np.float64)
        shrink = 0.9 if init_shrink is None else float(init_shrink)
    elif anchor_mode == "hold_xy":
        anchor, shrink = st_like.pos.copy(), 1.0
    else:
        anchor, shrink = np.array([0.0, 0.0, st_like.pos[2]], np.float64), (0.9 if init_shrink is None else float(init_shrink))

    # 环境 reset 后，进行若干次“注入 + 探测 step”
    obs = env.reset()
    ensure_env_double(env)
    last_obs = obs

    for k in range(max_tries):
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

        shrink *= float(C.REENTRY_SHRINK_FACTOR)  # 缩小位移再试

    print("[RE-ENTRY] 尝试用尽，使用最后注入位置进入。")
    return last_obs


def _done_reason(env, st_after_step: Optional[DroneState]) -> str:
    """
    粗略判断 done 的原因：
    - 若 env.t 接近 env.max_time -> TIME_LIMIT
    - 若位置超出世界边界（考虑 0.2 的缓冲）-> WORLD_BOUNDARY
    - 否则 -> UNKNOWN
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
        if np.any(np.abs(st.pos) > (C.WORLD_HALF_EXTENT - 0.2)):
            return "WORLD_BOUNDARY"
    except Exception:
        pass
    return "UNKNOWN"


# ==================================================
# 随机航点生成
# ==================================================

def gen_random_wp_xy(min_xy: float, max_xy: float) -> Tuple[float, float]:
    x = float(np.random.uniform(min_xy, max_xy))
    y = float(np.random.uniform(min_xy, max_xy))
    return x, y


def gen_random_wp() -> np.ndarray:
    """
    生成一个 3D 航点 [x,y,z]：
    """
    xy_lim = C.WORLD_HALF_EXTENT - C.XY_LIM_INSET
    x, y = gen_random_wp_xy(-xy_lim, xy_lim)
    z = float(np.random.uniform(C.Z_MIN, C.Z_MAX))
    return np.array([x, y, z], dtype=np.float64)


# ==================================================
# 主流程
# ==================================================

def main():
    """
    1) 创建并 reset 环境；
    2) while True：逐个生成随机航点并跟踪；
    3) 记录/打印每一阶段（追踪/到达/超时/重入/悬停）；
    4) 结束后统一输出 CSV、图像与动画。
    """
    print("[START] main_random_wps")

    # 1) 初始状态：起点设为原点（0,0,0），速度为 0
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
    obs = env.reset()
    ensure_env_double(env)

    # 3) 初始状态
    st0 = _get_state_from_obs(obs, 0)
    print(f"[INFO] 初始状态: pos={st0.pos}, vel={st0.vel}, |v|={speed_norm(st0.vel):.3f} m/s")

    # 4) 日志与绘图
    rows: List[List[str]] = []          
    xs: List[float] = []                # 轨迹 X 序列
    ys: List[float] = []                # 轨迹 Y 序列
    zs: List[float] = []                # 轨迹 Z 序列

    def log_state(t_s: float, phase: str, st: DroneState, wp=None, v_cmd=None):
        """
        把当前状态/指令写入 rows。
        - phase：阶段标记，如 "TRACK_WP_3" / "REACH_WP_3" / "TIMEOUT_WP_3" / "HOVER_WP_3" 等，便于 CSV 后处理。
        - speed：即时速度模长，便于观察速度变化。
        - wp_* ：当前目标航点（为空字符串表示本步没有目标，例如最终悬停）。
        - v_cmd_*：本步发送的速度指令（为空字符串表示本步未发指令或为 0）。
        """
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
        ])

    # 5) 主循环：按固定步长推进仿真时间
    DT = C.DT_GLOBAL  # 时间步长（秒）
    t_s = 0.0         # 仿真时间（秒，累加 DT）

    rand_waypoints: List[np.ndarray] = []  
    wp_count = 0                           # 航点计数

    while True:
        # 5.1) 终止条件：达到目标数量则 break
        if wp_count >= C.NUM_RANDOM_WP:
            print(f"[STOP] 已完成 {C.NUM_RANDOM_WP} 个随机航点，退出。")
            break

        # 5.2) 生成新的随机航点，并记录到列表（用于最终可视化）
        wp = gen_random_wp()
        rand_waypoints.append(wp.copy())
        wp_count += 1
        print(f"[NAV] 追踪随机航点 {wp_count}/{C.NUM_RANDOM_WP}: {wp.tolist()}")

        # 是否成功到达、重入计数
        reached = False
        reentry_count = 0
        t_wp_start = t_s
        t_last_reentry: Optional[float] = None

        step = 0  # 该航点下的 step 计数，用于控制打印频率

        # 5.3) 到达/超时/被跳过
        while True:
            # ---- 读状态 & 轨迹缓存 ----
            st = _get_state_from_obs(obs, 0)
            xs.append(float(st.pos[0])); ys.append(float(st.pos[1])); zs.append(float(st.pos[2]))

            # ---- 到达判定：距离到达阈值 WP_TOL 内即算成功 ----
            dist = float(np.linalg.norm(wp - st.pos))
            if dist <= C.WP_TOL:
                print(f"[OK] 已到达随机航点#{wp_count}  dist={dist:.3f}  pos={st.pos}")
                log_state(t_s, f"REACH_WP_{wp_count}", st, wp=wp, v_cmd=np.array([0,0,0], np.float64))
                reached = True
                break

            # ---- 航点超时判定：超过 MAX_TIME_PER_WP 秒未到达则跳过 ----
            elapsed = t_s - t_wp_start
            if elapsed >= C.MAX_TIME_PER_WP:
                print(f"[SKIP] 随机航点#{wp_count} 超时未到达（{elapsed:.2f}s ≥ {C.MAX_TIME_PER_WP:.2f}s），跳过。")
                log_state(t_s, f"TIMEOUT_WP_{wp_count}", st, wp=wp, v_cmd=np.array([0,0,0], np.float64))
                break

            # ---- 产生速度指令并前进一步 ----
            v_cmd = p_vel_to_waypoint(st.pos, wp, kp=C.KP, vmax=C.VMAX)
            _n = float(np.linalg.norm(v_cmd))
            if (dist > C.KICK_DIST_FACTOR * C.WP_TOL) and (_n < C.VMIN_KICK) and (_n > 1e-9):    
                    v_cmd = v_cmd * (C.VMIN_KICK / _n)  # 保持方向不变，将速度幅值提高到 VMIN_KICK

            ensure_env_double(env)
            obs, _r, done, _i = env.step(v_cmd.reshape(1,3))

            # ---- 记录本步（TRACK 阶段） ----
            log_state(t_s, f"TRACK_WP_{wp_count}", st, wp=wp, v_cmd=v_cmd)

            # ---- 打印运行状态（每 10 步打印一次，频繁打印会拖慢渲染/动画） ----
            if step % 10 == 0:
                print(f"[RUN] rand-wp#{wp_count} t={t_s:5.2f}s  pos={st.pos}  dist={dist:.3f}  v_cmd={v_cmd}  |v|={speed_norm(st.vel):.3f}  elapsed={elapsed:.2f}s")

            # ---- 推进时间与步计数 ----
            t_s += DT
            step += 1

            # ---- done 处理：触发“重入”机制 ----
            if _any_done(done):
                # ① 诊断 done 原因（可选打印，仅供排查）
                reason = _done_reason(env, _get_state_from_obs(obs, 0) if isinstance(obs,(dict,np.ndarray)) else None)

                # ② 保护窗：刚刚做过重入的 MIN_RUN_AFTER_REENTRY 时间窗口内，如果再次 done，
                #    不累计重入次数（short=True），防止边界条件导致无意义的“连环重入”。
                short = (t_last_reentry is not None) and ((t_s - t_last_reentry) < C.MIN_RUN_AFTER_REENTRY)
                if not short:
                    reentry_count += 1

                print(f"[DONE] rand-WP#{wp_count} 触发 done（reason={reason}），重入第 {reentry_count} 次（保护窗内: {short}）。 pos={st.pos}")

                # ③ 执行重入：以当前状态 st 为参考，选择“朝向航点”的锚定方式
                obs = _safe_reset_and_inject_with_probe(
                    env, st, wp_anchor_xy=wp, max_tries=8, anchor_mode="toward_wp", init_shrink=C.REENTRY_INIT_SHRINK
                )
                t_last_reentry = t_s

                # ④ 若重入次数超过上限，直接跳过此航点
                if (reentry_count > C.MAX_REENTRY_PER_WP) and not short:
                    print(f"[SKIP] rand-WP#{wp_count} 重入次数超限（{reentry_count}>{C.MAX_REENTRY_PER_WP}），跳过。")
                    log_state(t_s, f"SKIP_WP_{wp_count}", st, wp=wp, v_cmd=np.array([0,0,0], np.float64))
                    break

            # ---- 控制“仿真墙钟速度”：便于视觉观察。若追求纯计算速度可注释掉 ----
            time.sleep(DT)

        # 5.4) 到达后的短暂悬停
        if reached and C.HOVER_AFTER_REACH > 0.0:
            for _ in range(max(1, int(C.HOVER_AFTER_REACH / C.DT_GLOBAL))):
                st_h = _get_state_from_obs(obs, 0)
                ensure_env_double(env)
                obs, _r, done, _i = env.step(np.zeros((1,3), np.float64))

                if _any_done(done):
                    # 悬停中若触发 done，则在原地重入（hold_xy），尽量把飞机放回到当前位置
                    log_state(t_s, f"HOVER_WP_{wp_count}_DONE", st_h, wp=wp, v_cmd=np.array([0,0,0], np.float64))
                    print(f"[DONE] 悬停阶段触发 done（reason={_done_reason(env, None)}），原地重入继续。 pos={st_h.pos}")
                    obs = _safe_reset_and_inject_with_probe(env, st_h, wp_anchor_xy=None, max_tries=8, anchor_mode="hold_xy")
                    break

                # 正常悬停的记录
                log_state(t_s, f"HOVER_WP_{wp_count}", st_h, wp=wp, v_cmd=np.array([0,0,0], np.float64))
                xs.append(float(st_h.pos[0])); ys.append(float(st_h.pos[1])); zs.append(float(st_h.pos[2]))

                t_s += DT
                time.sleep(DT)

    # 6) 统一输出：CSV  平面图 3D 动画
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

    # 7) 结束状态
    st_end = _get_state_from_obs(obs, 0)
    print(f"[EXIT] 结束: pos={st_end.pos}, vel={st_end.vel}, |v|={speed_norm(st_end.vel):.3f} m/s")


if __name__ == "__main__":
    print("[BOOT] 程序启动")
    try:
        main()
    finally:
        print("[SHUTDOWN] 程序退出")
