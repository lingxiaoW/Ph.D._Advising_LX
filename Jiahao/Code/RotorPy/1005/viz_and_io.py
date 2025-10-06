
"""
平面图 3D动画 保存 CSV。
"""
from typing import List
import os
import numpy as np

def save_csv(rows: List[List[str]], csv_path: str) -> None:
    import csv
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "time_s","phase",
            "x","y","z","vx","vy","vz","speed",
            "wp_x","wp_y","wp_z","v_cmd_x","v_cmd_y","v_cmd_z"
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

    # 3D 轨迹
    fig = plt.figure(figsize=(6.08, 6.08), dpi=100)
    ax = fig.add_subplot(111, projection='3d')  # type: ignore
    ax.plot(xs, ys, zs, label='trajectory')
    ax.scatter(xs[0], ys[0], zs[0], marker='o', s=40, label='start')
    ax.scatter(xs[-1], ys[-1], zs[-1], marker='^', s=40, label='end')
    if waypoints:
        wps = np.stack(waypoints, 0)
        ax.scatter(wps[:,0], wps[:,1], wps[:,2], marker='x', s=50, label='waypoints')
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z'); ax.set_title('3D Trajectory')
    try: ax.set_box_aspect((1,1,1))  # type: ignore
    except Exception: pass
    ax.legend(); fig.tight_layout(); fig.savefig(fig_traj_3d, dpi=150); plt.close(fig)
    print(f"[SAVE] 3D 轨迹图：{fig_traj_3d}")

    # XY 俯视
    fig = plt.figure(figsize=(6.08, 6.08), dpi=100)
    ax = fig.add_subplot(111)
    ax.plot(xs, ys, label='trajectory')
    ax.scatter(xs[0], ys[0], marker='o', s=40, label='start')
    ax.scatter(xs[-1], ys[-1], marker='^', s=40, label='end')
    if waypoints:
        wps = np.stack(waypoints, 0)
        ax.scatter(wps[:,0], wps[:,1], s=50, marker='x', label='waypoints')
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_title('XY Top-Down')
    ax.axis('equal'); ax.grid(True); ax.legend()
    fig.tight_layout(); fig.savefig(fig_traj_xy, dpi=150); plt.close(fig)
    print(f"[SAVE] XY 俯视图：{fig_traj_xy}")

    # 高度-时间
    t = np.arange(len(zs))*dt
    fig = plt.figure(figsize=(7,3.5)); ax = fig.add_subplot(111)
    ax.plot(t, zs, label='z(t)'); ax.set_xlabel('Time [s]'); ax.set_ylabel('Z [m]'); ax.grid(True)
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
        print("[WARN] 3D 动画跳过（缺少依赖）：", e); return

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
    ax = fig.add_subplot(111, projection='3d')  # type: ignore
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
        ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_zlim(*zlim)
        try: ax.set_box_aspect((1, 1, 1))  # type: ignore
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
