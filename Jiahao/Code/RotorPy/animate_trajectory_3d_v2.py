
# animate_trajectory_3d_v2.py
# Same as v1 but properly clears previous quiver arrows each frame to avoid build-up.

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

def set_axes_equal(ax):
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)
    plot_radius = 0.5*max([x_range, y_range, z_range])
    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

def main(csv_path="uav_telemetry.csv", out_mp4="traj3d.mp4", out_gif="traj3d.gif",
         fps=30, max_frames=1200, show_velocity=False, show_heading=False):
    df = pd.read_csv(csv_path)
    required = ["time","x","y","z"]
    miss = [c for c in required if c not in df.columns]
    if miss:
        raise RuntimeError(f"CSV missing columns: {miss}")

    t = df["time"].to_numpy()
    x = df["x"].to_numpy()
    y = df["y"].to_numpy()
    z = df["z"].to_numpy()

    yaw = df["yaw_deg"].to_numpy() if "yaw_deg" in df.columns else None
    vx = df["vx"].to_numpy() if "vx" in df.columns else None
    vy = df["vy"].to_numpy() if "vy" in df.columns else None
    vz = df["vz"].to_numpy() if "vz" in df.columns else None

    N = len(t)
    if N < 2:
        raise RuntimeError("Not enough samples for animation.")

    frames = np.linspace(0, N-1, min(N, max_frames), dtype=int)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("UAV 3D Trajectory")

    ax.scatter([x[0]], [y[0]], [z[0]], marker='o', label="start")
    ax.scatter([x[-1]], [y[-1]], [z[-1]], marker='^', label="end")

    line, = ax.plot([], [], [], lw=2, label="path")
    point, = ax.plot([], [], [], marker='o', label="current")

    # Keep references to current quivers to remove them each frame
    artists_to_remove = []

    pad = 0.1
    xmin, xmax = np.min(x), np.max(x)
    ymin, ymax = np.min(y), np.max(y)
    zmin, zmax = np.min(z), np.max(z)
    dx = xmax - xmin
    dy = ymax - ymin
    dz = zmax - zmin
    if dx == dy == dz == 0:
        dx = dy = dz = 1.0
    ax.set_xlim(xmin - pad*dx, xmax + pad*dx)
    ax.set_ylim(ymin - pad*dy, ymax + pad*dy)
    ax.set_zlim(zmin - pad*dz, zmax + pad*dz)
    set_axes_equal(ax)
    ax.legend(loc="best")

    def init():
        line.set_data([], [])
        line.set_3d_properties([])
        point.set_data([], [])
        point.set_3d_properties([])
        return line, point

    def update(idx):
        # remove previous dynamic artists (quivers) to avoid buildup
        nonlocal artists_to_remove
        for art in artists_to_remove:
            try:
                art.remove()
            except Exception:
                pass
        artists_to_remove = []

        i = frames[idx]
        line.set_data(x[:i+1], y[:i+1])
        line.set_3d_properties(z[:i+1])
        point.set_data([x[i]], [y[i]])
        point.set_3d_properties([z[i]])

        # Velocity vector
        if show_velocity and vx is not None and vy is not None and vz is not None:
            vscale = 0.5
            qv = ax.quiver(x[i], y[i], z[i], vx[i], vy[i], vz[i], length=vscale, normalize=True)
            artists_to_remove.append(qv)

        # Heading arrow in XY plane using yaw
        if show_heading and yaw is not None and not np.isnan(yaw[i]):
            rad = math.radians(yaw[i])
            hx = math.cos(rad); hy = math.sin(rad)
            hscale = 0.5
            qh = ax.quiver(x[i], y[i], z[i], hx, hy, 0.0, length=hscale, normalize=True)
            artists_to_remove.append(qh)

        return [line, point] + artists_to_remove

    total_frames = len(frames)
    anim = animation.FuncAnimation(
        fig, update, init_func=init, frames=total_frames, interval=1000/fps, blit=False
    )

    saved = False
    try:
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=fps, metadata=dict(artist='animate_trajectory_3d_v2'), bitrate=1800)
        anim.save(out_mp4, writer=writer)
        print(f"Saved MP4: {out_mp4}")
        saved = True
    except Exception as e:
        print("MP4 save failed (ffmpeg missing?). Error:", e)

    if not saved:
        try:
            from matplotlib.animation import PillowWriter
            anim.save(out_gif, writer=PillowWriter(fps=fps))
            print(f"Saved GIF: {out_gif}")
            saved = True
        except Exception as e:
            print("GIF save failed. Error:", e)
            raise

if __name__ == "__main__":
    main()
