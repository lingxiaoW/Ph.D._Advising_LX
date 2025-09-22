
# drone_quickstart_en_v5.py
# ------------------------------------------------------------
# RotorPy quickstart script (English, well-commented)
# - Uses a Lissajous trajectory in XY with constant Z "height"
# - Adds a sinusoidal wind field (vector form)
# - Runs the simulation and extracts results in a version-robust way
# - Exports key telemetry to CSV (time, position, velocity, yaw/pitch/roll)
# - Builds an inline 3D animation similar to the official demo (MP4/GIF)
# ------------------------------------------------------------

import math
import numpy as np

# --- RotorPy components (imports may vary with versions) ---
from rotorpy.environments import Environment
from rotorpy.vehicles.multirotor import Multirotor
from rotorpy.vehicles.crazyflie_params import quad_params  # dict-like parameter set
from rotorpy.controllers.quadrotor_control import SE3Control
from rotorpy.trajectories.lissajous_traj import TwoDLissajous
from rotorpy.wind.default_winds import SinusoidWind


# ------------------------ Math helpers ------------------------
def quat_to_euler_zyx(q):
    """
    Convert quaternion [i, j, k, w] to Euler angles (ZYX) in radians.
    Returns (yaw, pitch, roll).
    """
    qi, qj, qk, qw = q
    # yaw (Z)
    siny_cosp = 2.0 * (qw*qk + qi*qj)
    cosy_cosp = 1.0 - 2.0 * (qj*qj + qk*qk)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    # pitch (Y)
    sinp = 2.0 * (qw*qj - qk*qi)
    if abs(sinp) >= 1:
        pitch = math.copysign(math.pi/2, sinp)  # clamp at +/-90 deg if needed
    else:
        pitch = math.asin(sinp)
    # roll (X)
    sinr_cosp = 2.0 * (qw*qi + qj*qk)
    cosr_cosp = 1.0 - 2.0 * (qi*qi + qj*qk)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    return yaw, pitch, roll


# --------------------- Result extraction ----------------------
def _get_val(d, key):
    """Safe dict get: only returns value if d is a dict and key exists; otherwise None."""
    if isinstance(d, dict) and key in d:
        return d[key]
    return None

def _first_non_none(*vals):
    """Return the first value that is not None (avoid using 'or' on NumPy arrays)."""
    for v in vals:
        if v is not None:
            return v
    return None

def extract_results(results):
    """
    Robustly extract (t, pos, vel, quat) from a RotorPy result dict across versions.
    Many versions differ in key names and nesting; we try multiple aliases.
      - t/time: (N,)
      - pos: (N,3) or reconstructed from x,y,z
      - vel: (N,3) or reconstructed from vx,vy,vz
      - quat: optional (N,4)
    Raises RuntimeError if essential fields are missing.
    """
    import numpy as np

    container = _get_val(results, 'state') or results

    # time
    t = _first_non_none(
        _get_val(container, 't'),
        _get_val(results, 't'),
        _get_val(container, 'time'),
        _get_val(results, 'time'),
    )

    # position
    pos = _first_non_none(
        _get_val(container, 'x'),
        _get_val(results, 'x'),
        _get_val(container, 'pos'),
        _get_val(results, 'pos'),
    )
    if pos is None:
        xs = _first_non_none(_get_val(container, 'x'), _get_val(container, 'x_m'), _get_val(container, 'x_pos'))
        ys = _first_non_none(_get_val(container, 'y'), _get_val(container, 'y_m'), _get_val(container, 'y_pos'))
        zs = _first_non_none(_get_val(container, 'z'), _get_val(container, 'z_m'), _get_val(container, 'z_pos'))
        if xs is not None and ys is not None and zs is not None:
            pos = np.column_stack([xs, ys, zs])

    # velocity
    vel = _first_non_none(
        _get_val(container, 'v'),
        _get_val(results, 'v'),
        _get_val(container, 'vel'),
        _get_val(results, 'vel'),
    )
    if vel is None:
        vxs = _first_non_none(_get_val(container, 'vx'), _get_val(container, 'vx_m'), _get_val(container, 'x_dot'))
        vys = _first_non_none(_get_val(container, 'vy'), _get_val(container, 'vy_m'), _get_val(container, 'y_dot'))
        vzs = _first_non_none(_get_val(container, 'vz'), _get_val(container, 'vz_m'), _get_val(container, 'z_dot'))
        if vxs is not None and vys is not None and vzs is not None:
            vel = np.column_stack([vxs, vys, vzs])

    # quaternion (optional)
    quat = _first_non_none(
        _get_val(container, 'q'),
        _get_val(results, 'q'),
        _get_val(container, 'quat'),
        _get_val(results, 'quat'),
    )

    missing = []
    if t is None: missing.append('t/time')
    if pos is None: missing.append('x/pos or x,y,z')
    if vel is None: missing.append('v/vel or vx,vy,vz')
    if missing:
        print('[DEBUG] results.keys():', list(results.keys()) if isinstance(results, dict) else type(results))
        if isinstance(container, dict):
            try:
                print('[DEBUG] (state).keys():', list(container.keys()))
            except Exception:
                pass
        raise RuntimeError('Missing required result fields: ' + ', '.join(missing))

    return t, pos, vel, quat


# --------------------- 3D animation utilities ---------------------
def _set_axes_equal(ax):
    """Set equal scaling for 3D axes so that X, Y, Z units are comparable."""
    import numpy as np
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
    x_range = abs(x_limits[1] - x_limits[0]); x_mid = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0]); y_mid = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0]); z_mid = np.mean(z_limits)
    r = 0.5 * max([x_range, y_range, z_range])
    ax.set_xlim3d([x_mid - r, x_mid + r])
    ax.set_ylim3d([y_mid - r, y_mid + r])
    ax.set_zlim3d([z_mid - r, z_mid + r])


def save_animation_3d(t, pos, vel=None, quat=None, out_mp4="sim_anim.mp4",
                      out_gif="sim_anim.gif", fps=30, max_frames=1200,
                      show_heading=False, show_velocity=False):
    """
    Create a simple 3D animation:
      - A growing trajectory line (X,Y,Z) and a current-position marker
      - Optional: velocity vector and heading arrow (from yaw)
      - Saves MP4 if ffmpeg is available, otherwise saves GIF
    """
    import math
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import animation

    # Ensure shapes
    t = np.asarray(t).reshape(-1)
    pos = np.asarray(pos).reshape(len(t), -1)
    if pos.shape[1] < 3:
        raise RuntimeError("pos must have shape (N, 3)")

    x, y, z = pos[:,0], pos[:,1], pos[:,2]

    # Optional velocity
    vx = vy = vz = None
    if vel is not None:
        vel = np.asarray(vel).reshape(len(t), -1)
        if vel.shape[1] >= 3:
            vx, vy, vz = vel[:,0], vel[:,1], vel[:,2]

    # Optional heading (from quaternion -> yaw deg)
    yaw_deg = None
    if quat is not None:
        quat = np.asarray(quat).reshape(len(t), -1)
        if quat.shape[1] == 4:
            yaw_list = []
            for q in quat:
                yaw, pitch, roll = quat_to_euler_zyx(q)
                yaw_list.append(math.degrees(yaw))
            yaw_deg = np.asarray(yaw_list)

    # Subsample frames for long sequences
    N = len(t)
    frames = np.linspace(0, N-1, min(N, max_frames), dtype=int)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("UAV Simulation (3D Trajectory)")

    # Axis limits with padding and equal scaling
    pad = 0.1
    xmin, xmax = float(np.min(x)), float(np.max(x))
    ymin, ymax = float(np.min(y)), float(np.max(y))
    zmin, zmax = float(np.min(z)), float(np.max(z))
    dx, dy, dz = xmax - xmin, ymax - ymin, zmax - zmin
    if dx == dy == dz == 0:
        dx = dy = dz = 1.0
    ax.set_xlim(xmin - pad*dx, xmax + pad*dx)
    ax.set_ylim(ymin - pad*dy, ymax + pad*dy)
    ax.set_zlim(zmin - pad*dz, zmax + pad*dz)
    _set_axes_equal(ax)

    # Start/end markers
    ax.scatter([x[0]], [y[0]], [z[0]], marker='o', label="start")
    ax.scatter([x[-1]], [y[-1]], [z[-1]], marker='^', label="end")

    # Trajectory line (growing) + current point
    line, = ax.plot([], [], [], lw=2, label="path")
    point, = ax.plot([], [], [], marker='o', label="current")

    # Temporary artists that are cleaned every frame (e.g., quiver arrows)
    dynamic_artists = []
    ax.legend(loc="best")

    def init():
        line.set_data([], [])
        line.set_3d_properties([])
        point.set_data([], [])
        point.set_3d_properties([])
        return [line, point]

    def update(frame_idx):
        nonlocal dynamic_artists
        # Remove arrows from previous frame to avoid "blue buildup"
        for art in dynamic_artists:
            try:
                art.remove()
            except Exception:
                pass
        dynamic_artists = []

        i = frames[frame_idx]
        line.set_data(x[:i+1], y[:i+1])
        line.set_3d_properties(z[:i+1])
        point.set_data([x[i]], [y[i]])
        point.set_3d_properties([z[i]])

        # Optional: velocity vector
        if False and show_velocity and vx is not None:
            vscale = 0.5
            qv = ax.quiver(x[i], y[i], z[i], vx[i], vy[i], vz[i], length=vscale, normalize=True)
            dynamic_artists.append(qv)

        # Optional: heading arrow from yaw (XY plane)
        if False and show_heading and yaw_deg is not None and not np.isnan(yaw_deg[i]):
            rad = math.radians(float(yaw_deg[i]))
            hx, hy = math.cos(rad), math.sin(rad)
            hscale = 0.5
            qh = ax.quiver(x[i], y[i], z[i], hx, hy, 0.0, length=hscale, normalize=True)
            dynamic_artists.append(qh)

        return [line, point] + dynamic_artists

    anim = animation.FuncAnimation(fig, update, init_func=init,
                                   frames=len(frames), interval=1000/fps, blit=False)

    # Try MP4 first (requires ffmpeg). Fallback to GIF if unavailable.
    saved = False
    try:
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=fps, metadata=dict(artist='drone_quickstart_en_v5'), bitrate=1800)
        anim.save(out_mp4, writer=writer)
        print(f"[Saved] MP4 animation -> {out_mp4}")
        saved = True
    except Exception as e:
        print("[Warn] MP4 export failed (ffmpeg missing?).", e)

    if not saved:
        try:
            from matplotlib.animation import PillowWriter
            anim.save(out_gif, writer=PillowWriter(fps=fps))
            print(f"[Saved] GIF animation -> {out_gif}")
            saved = True
        except Exception as e:
            print("[Error] GIF export failed.", e)
            raise


# ----------------------------- Main -----------------------------
def main():
    print("=== RotorPy UAV Quickstart (English v5) ===")

    # --- 1) Build system components ---
    vehicle = Multirotor(quad_params)     # Crazyflie-like parameters
    controller = SE3Control(quad_params)  # Simple SE(3) position controller

    # Trajectory: Lissajous in XY (A/B amplitudes, a/b frequencies) with constant Z "height"
    traj = TwoDLissajous(A=0.5, B=0.5, a=0.2, b=0.25, height=1.0)

    # Wind: sinusoidal, specified in vector form (amplitudes, frequencies, phases)
    # Here we construct horizontal wind from a scalar speed U and direction theta:
    U = 1.0                          # target horizontal wind speed [m/s]
    theta = np.deg2rad(30.0)         # wind direction [rad], measured from +X toward +Y
    T = 8.0                          # period [s]
    f_hz = 1.0 / T                   # frequency [Hz] (use 2*pi/T for angular frequency builds)

    amplitudes = np.array([U*np.cos(theta), U*np.sin(theta), 0.0])  # wind amplitude on X/Y/Z
    frequencies = np.array([f_hz, f_hz, 0.0])                       # per-axis frequency
    phase = np.zeros(3)                                              # phase offsets (radians)

    wind = SinusoidWind(amplitudes=amplitudes,
                        frequencies=frequencies,
                        phase=phase)

    # Environment ties everything together and runs the simulation
    env = Environment(
        vehicle=vehicle,
        controller=controller,
        trajectory=traj,
        wind_profile=wind,
        sim_rate=100,            # simulation step rate [Hz]
        imu=None, mocap=None, estimator=None, world=None,
        safety_margin=0.25
    )

    # Optional: initial state near hover (adjust as needed)
    x0 = {
        'x': np.array([0.0, 0.0, 0.0]),              # position [m]
        'v': np.zeros(3),                            # linear velocity [m/s]
        'q': np.array([0.0, 0.0, 0.0, 1.0]),         # quaternion [i, j, k, w]
        'w': np.zeros(3),                            # body angular velocity [rad/s]
        'wind': np.array([0.0, 0.0, 0.0]),           # initial wind sample
        'rotor_speeds': np.array([1788.5]*4)         # initial rotor speeds (example)
    }
    env.vehicle.initial_state = x0

    # --- 2) Run simulation ---
    import inspect
    try:
        run_sig = inspect.signature(env.run)
        if 'fname' in run_sig.parameters:
            results = env.run(
                t_final=10.0,
                use_mocap=False,
                terminate=False,
                plot=False, plot_mocap=False, plot_estimator=False, plot_imu=False,
                animate_bool=True, animate_wind=False,   # enable built-in animation
                verbose=True, fname="builtin_anim.mp4"   # save if supported
            )
        else:
            results = env.run(
                t_final=10.0,
                use_mocap=False,
                terminate=False,
                plot=False, plot_mocap=False, plot_estimator=False, plot_imu=False,
                animate_bool=True, animate_wind=False,
                verbose=True
            )
    except TypeError:
        # Some very old signatures may not accept some flags; fall back to essentials
        results = env.run(
            t_final=10.0,
            animate_bool=True,
        )

    # --- 3) Extract data in a version-robust way ---
    t, pos, vel, quat = extract_results(results)
    N = len(t)
    freq = N / (t[-1] - t[0] + 1e-9)
    print(f"\n[Summary] Collected {N} samples @ ~{freq:.1f} Hz")

    # Print a brief 10 Hz telemetry for sanity check
    sample_rate_hz = 10
    sim_rate = 100
    step = max(1, sim_rate // sample_rate_hz)
    print("\n[Telemetry @ ~10 Hz]")
    print("time   | position (m)         | speed (m/s) | yaw/pitch/roll (deg)")
    print("-------+-----------------------+-------------+----------------------")
    for i in range(0, N, step):
        px, py, pz = pos[i]
        vx, vy, vz = vel[i]
        speed = float(np.linalg.norm(vel[i]))
        if quat is not None:
            yaw, pitch, roll = quat_to_euler_zyx(quat[i])
            ypr_deg = (math.degrees(yaw), math.degrees(pitch), math.degrees(roll))
        else:
            ypr_deg = (float('nan'), float('nan'), float('nan'))
        print(f"{t[i]:5.2f} | [{px:6.3f} {py:6.3f} {pz:6.3f}] | {speed:6.3f}     | "
              f"[{ypr_deg[0]:6.1f} {ypr_deg[1]:6.1f} {ypr_deg[2]:6.1f}]")

    print("\n[Final] pos(m):", pos[-1], "vel(m/s):", vel[-1])
    if quat is not None:
        ypr = tuple(math.degrees(a) for a in quat_to_euler_zyx(quat[-1]))
        print("yaw/pitch/roll (deg):", ypr)
    else:
        print("yaw/pitch/roll: N/A (no quaternion in results)")

    # --- 4) Save CSV for downstream analysis ---
    import csv
    csv_path = "uav_telemetry.csv"
    if quat is not None:
        ypr_deg_arr = np.vstack([[math.degrees(a) for a in quat_to_euler_zyx(q)] for q in quat])
    else:
        ypr_deg_arr = np.full((len(t), 3), np.nan, dtype=float)
    pos_arr = np.asarray(pos).reshape(len(t), -1)
    vel_arr = np.asarray(vel).reshape(len(t), -1)
    header = ["time","x","y","z","vx","vy","vz","yaw_deg","pitch_deg","roll_deg"]
    rows = np.column_stack([t, pos_arr, vel_arr, ypr_deg_arr])
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows.tolist())
    print(f"\n[Saved] CSV written to {csv_path} with {len(rows)} rows.")

    # --- 5) Create a 3D animation similar to the official demo ---
    try:
        save_animation_3d(t, pos, vel, quat,
                          out_mp4="sim_anim.mp4", out_gif="sim_anim.gif",
                          fps=30, max_frames=1200,
                          show_heading=False, show_velocity=False)
    except Exception as e:
        print("[Warn] Animation generation failed:", e)


if __name__ == "__main__":
    main()
