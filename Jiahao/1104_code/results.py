# -*- coding: utf-8 -*-
from __future__ import annotations
import csv
from typing import Dict, List, Tuple, Sequence, Optional
import numpy as np


# CSV writer (robust columns)

def write_csv(path: str,
              rows: List[Dict[str, float]],
              field_order: Optional[Sequence[str]] = None) -> None:
    """
    """
    if not rows:
        return

    if field_order is None:
        
        keyset = set()
        for r in rows:
            keyset |= set(r.keys())
        
        priority = ["t","x","y","z","speed","heading_deg","c","wind_v","wind_phi","wp_idx","dist","status"]
        ordered = [k for k in priority if k in keyset]
        ordered += [k for k in sorted(keyset) if k not in ordered]
        field_order = ordered

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(field_order))
        w.writeheader()
        for r in rows:
            w.writerow(r)



# 3D plot 

def plot_3d(path_xyz: Sequence[Tuple[float, float, float]],
            waypoints: Sequence[np.ndarray],
            bounds,  # (xmin,xmax,ymin,ymax,zmin,zmax)
            path: str,
            show: bool=False,
            c_list: Optional[Sequence[float]] = None,  
            wind_vec: Optional[Tuple[float,float]] = None  
            ):
    """
    3D 轨迹图：
    """
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        xs, ys, zs = zip(*path_xyz)
        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot(111, projection='3d')

        if c_list is not None and len(c_list) == len(xs):
        
            c_vals = np.asarray(c_list, dtype=float)
           
            cmin, cmax = np.nanpercentile(c_vals, [1, 99])
            if not np.isfinite(cmin) or not np.isfinite(cmax) or cmax <= cmin:
                cmin, cmax = float(np.nanmin(c_vals)), float(np.nanmax(c_vals))
            if cmax <= cmin:
                cmin, cmax = 0.0, 1.0
            cnorm = (c_vals - cmin) / (cmax - cmin + 1e-12)
           
            ax.scatter(xs, ys, zs, c=cnorm, cmap="viridis", s=8, label='trajectory (colored by conc)')
        else:
            ax.plot(xs, ys, zs, linewidth=1.5, label='trajectory')

        ax.scatter(xs[0], ys[0], zs[0], s=30, marker='^', label='start')

        wpx = [float(wp[0]) for wp in waypoints]
        wpy = [float(wp[1]) for wp in waypoints]
        wpz = [float(wp[2]) for wp in waypoints]
        ax.scatter(wpx, wpy, wpz, s=25, marker='o', label='waypoints')

        xmin,xmax,ymin,ymax,zmin,zmax = bounds
        ax.set_title('UAV Trajectory (No-Gym)')
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_zlabel('Z [m]')
        ax.set_xlim([xmin, xmax])
        ax.set_ylim([ymin, ymax])
        ax.set_zlim([zmin, zmax])

        
        if wind_vec is not None:
            Ux, Uy = wind_vec
            ax.quiver(xmin + 0.08*(xmax-xmin),
                      ymin + 0.08*(ymax-ymin),
                      zmin + 0.8*(zmax-zmin),
                      Ux, Uy, 0.0, length=0.15*(xmax-xmin), normalize=True, color="k")
            ax.text(xmin + 0.08*(xmax-xmin),
                    ymin + 0.08*(ymax-ymin),
                    zmin + 0.8*(zmax-zmin),
                    f"Wind ({Ux:.1f},{Uy:.1f}) m/s", fontsize=8)

        ax.legend(loc='upper left')
        fig.tight_layout()

        fig.savefig(path, dpi=160)
        if show:
            plt.show()
        else:
            plt.close(fig)
        print(f"[SAVE] Plot -> {path}")
    except Exception as e:
        print(f"[WARN] 绘图失败: {e}")


# --------------------------------------
# Time-series plots (NEW)
# --------------------------------------
def plot_timeseries(t_list: Sequence[float],
                    series: Dict[str, Sequence[float]],
                    out_png: str,
                    show: bool=False):
    """
    """
    try:
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(9,5))
        ax = fig.add_subplot(111)

        for name, vals in series.items():
            if vals is None: 
                continue
            if len(vals) != len(t_list):
                print(f"[WARN] Skip series {name}: length mismatch")
                continue
            ax.plot(t_list, vals, label=name, linewidth=1.2)

        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Value")
        ax.set_title("UAV Telemetry")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_png, dpi=160)
        if show:
            plt.show()
        else:
            plt.close(fig)
        print(f"[SAVE] Timeseries -> {out_png}")
    except Exception as e:
        print(f"[WARN] 时间序列绘图失败: {e}")
