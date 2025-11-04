import os
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  

Xmin, Xmax, Ymin, Ymax = -50.0, 300.0, -150.0, 150.0
dx = dy = 2.0
z_slice = 1.5

SRC = np.array([0.0, 0.0, 1.5])
Q_PUFF = 50.0
U_MEAN = 2.0

A_Y = 1.2
A_Z = 0.8

SIM_DT  = 1.0
PUFF_DT = 5.0
TOTAL_T = 6 * 60.0
PERC_FOR_VMAX = 99.5

xs = np.arange(Xmin, Xmax + dx, dx)
ys = np.arange(Ymin, Ymax + dy, dy)
X, Y = np.meshgrid(xs, ys, indexing="xy")
Z = np.full_like(X, z_slice)

def wind_vec_at(t: float):
    return float(U_MEAN), 0.0

puffs = []
t_since_emit = PUFF_DT

def cp_concentration(x, y, z, puff, Q):
    xp, yp, zp, age = puff
    if age <= 0.0:
        return np.zeros_like(x, dtype=np.float64)
    sig_y = max(1e-6, A_Y * math.sqrt(age))
    sig_z = max(1e-6, A_Z * math.sqrt(age))
    norm = Q / ((2.0 * math.pi) ** 1.5 * (sig_y ** 2) * sig_z)
    rx2 = (x - xp) ** 2 + (y - yp) ** 2
    horiz = np.exp(-0.5 * rx2 / (sig_y ** 2))
    vert = np.exp(-0.5 * ((z - zp) ** 2) / (sig_z ** 2)) + np.exp(-0.5 * ((z + zp) ** 2) / (sig_z ** 2))
    return norm * horiz * vert

def advect_and_age(puffs_list, dt, Ux, Uy):
    for p in puffs_list:
        p[0] += Ux * dt
        p[1] += Uy * dt
        p[3] += dt

def render_3d_slices_alpha(puffs_snapshot, t, Ux, Uy, zs=tuple(range(0,21,1)),
                           base_alpha=0.35, min_alpha=0.04, gamma=1.6, dpi=260):
    
    # 计算各z
    C_slices = []
    for z in zs:
        Zz = np.full_like(X, float(z))
        C = np.zeros_like(X, dtype=np.float64)
        for p in puffs_snapshot:
            C += cp_concentration(X, Y, Zz, p, Q_PUFF)
        C_slices.append((z, C))

    
    all_vals = np.concatenate([C.ravel() for _, C in C_slices])
    vmax = np.nanpercentile(all_vals, PERC_FOR_VMAX)
    vmax = float(vmax) if np.isfinite(vmax) and vmax > 0 else 1e-6

    import matplotlib.colors as mcolors
    vir = plt.cm.viridis(np.linspace(0, 1, 256))
    k = 32
    vir[:k, :3] = vir[:k, :3] * 0.3 + 0.7
    cmap = mcolors.ListedColormap(vir)

    from matplotlib.colors import Normalize
    norm = Normalize(vmin=0.0, vmax=vmax)

    fig = plt.figure(figsize=(11.5, 7.0), dpi=dpi)
    ax = fig.add_subplot(111, projection='3d')

    for z, C in C_slices:
        Zplane = np.full_like(X, float(z))
        facecolors = cmap(norm(C))
        a = norm(C)
        a = np.clip(a, 0.0, 1.0) ** float(gamma)
        a = min_alpha + (base_alpha - min_alpha) * a
        facecolors[..., -1] = a
        ax.plot_surface(X, Y, Zplane, rstride=1, cstride=1, facecolors=facecolors,
                        linewidth=0, antialiased=False, shade=False)

    import matplotlib as mpl
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label('concentration (arb. unit)')

    ax.scatter([float(SRC[0])], [float(SRC[1])], [float(SRC[2])], color='white', s=30, depthshade=False)
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_zlabel('z (m)')
    ax.set_xlim(Xmin, Xmax)
    ax.set_ylim(Ymin, Ymax)
    ax.set_zlim(min(zs), max(zs))
    ax.view_init(elev=25, azim=-60)
    ax.set_title(f"3D stacked slices (z=0..20m, step=1m) | t={int(t)} s | U=({Ux:.1f},{Uy:.1f}) m/s | puffs={len(puffs_snapshot)}")

    fig.tight_layout()
    out_name = f"puff_final_3d_slices_alpha_z0_20_step1_t{int(t)}.png"
    fig.savefig(out_name, dpi=dpi)
    plt.close(fig)

    print(f"[OK] Saved 3D alpha-stacked slices (21 layers): {os.path.abspath(out_name)}")


def simulate():
    global t_since_emit, puffs
    n_steps = int(TOTAL_T / SIM_DT)
    for step in range(n_steps):
        t = step * SIM_DT
        Ux, Uy = wind_vec_at(t)
        t_since_emit += SIM_DT
        if t_since_emit >= PUFF_DT:
            puffs.append([float(SRC[0]), float(SRC[1]), float(SRC[2]), 1e-6])
            t_since_emit = 0.0
        advect_and_age(puffs, SIM_DT, Ux, Uy)
        alive = [p for p in puffs if (Xmin - 50 <= p[0] <= Xmax + 50 and Ymin - 50 <= p[1] <= Ymax + 50)]
        puffs[:] = alive
        if step == n_steps - 1:
            puffs_snapshot = [p.copy() for p in puffs]
            render_3d_slices_alpha(puffs_snapshot, t, Ux, Uy, zs=tuple(range(0,21,1)))

if __name__ == '__main__':
    simulate()
