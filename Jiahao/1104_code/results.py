#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
results.py

保存与可视化模块
----------------
提供以下功能:
    - write_csv(rows, path)
    - plot_3d(path_xyz, src_list, bounds, path, c_list, wind_vec)
    - plot_timeseries(t_list, series, out_png)
"""

import csv
import math
from typing import List, Dict, Optional, Sequence, Tuple
import numpy as np
import matplotlib.pyplot as plt


# ===============================================================
# 1. CSV 写入函数
# ===============================================================
def write_csv(rows: List[Dict[str, float]],
              path: str,
              field_order: Optional[Sequence[str]] = None) -> None:
    """
    写入 CSV 文件。
    Parameters
    ----------
    rows : list of dict
        每行的数据字典，例如 {'t': 0.0, 'x': 1.2, 'y': 3.4, ...}
    path : str
        输出文件路径
    field_order : list of str, optional
        字段顺序（若不提供则自动推断）
    """

    # 防止主程序参数顺序写反
    if isinstance(rows, str) and isinstance(path, list):
        print("[WARN] Detected reversed write_csv args, auto-swapping them.")
        rows, path = path, rows

    if not rows:
        print("[WARN] write_csv: empty data, nothing to write.")
        return

    # 自动推断字段顺序
    if field_order is None:
        keyset = set()
        for r in rows:
            if isinstance(r, dict):
                keyset |= set(r.keys())
        priority = [
            "t", "x", "y", "z",
            "speed", "heading_deg",
            "c", "wind_v", "wind_phi",
            "dist_to_src", "wp_idx", "status"
        ]
        ordered = [k for k in priority if k in keyset]
        ordered += [k for k in sorted(keyset) if k not in ordered]
        field_order = ordered

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(field_order))
        writer.writeheader()
        for r in rows:
            if isinstance(r, dict):
                writer.writerow(r)

    print(f"[SAVE] CSV written -> {path} ({len(rows)} rows)")


# ===============================================================
# 2. 三维轨迹绘图
# ===============================================================
def plot_3d(path_xyz: List[Tuple[float, float, float]],
            src_list: List[np.ndarray],
            bounds: Tuple[float, float, float, float, float, float],
            path: str,
            c_list: Optional[List[float]] = None,
            wind_vec: Optional[Tuple[float, float]] = None) -> None:
    """
    绘制无人机三维轨迹图。
    """
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    X, Y, Z = zip(*path_xyz)
    if c_list is not None:
        sc = ax.scatter(X, Y, Z, c=c_list, cmap="viridis", s=8)
        plt.colorbar(sc, label="Gas Concentration")
    else:
        ax.plot(X, Y, Z, color="b", lw=1.5)

    for src in src_list:
        ax.scatter(src[0], src[1], src[2], color="r", s=60, label="Source")

    # 边界框
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_zlim(zmin, zmax)

    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.set_title("UAV Plume Tracing Trajectory")

    if wind_vec is not None:
        Ux, Uy = wind_vec
        ax.quiver(xmin + 20, ymin + 20, zmax - 1,
                  Ux, Uy, 0, length=30, color="cyan", label="Wind")

    ax.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[SAVE] 3D Trajectory Plot -> {path}")


# ===============================================================
# 3. 时间序列绘图
# ===============================================================
def plot_timeseries(t_list: List[float],
                    series: Dict[str, List[float]],
                    out_png: str) -> None:
    """
    绘制速度、风速、浓度等时间变化曲线。
    """
    fig, axes = plt.subplots(len(series), 1, figsize=(7, 8), sharex=True)
    if len(series) == 1:
        axes = [axes]

    for i, (k, v) in enumerate(series.items()):
        axes[i].plot(t_list, v, lw=1.2)
        axes[i].set_ylabel(k)
        axes[i].grid(True, linestyle="--", alpha=0.5)

    axes[-1].set_xlabel("Time [s]")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"[SAVE] Time Series Plot -> {out_png}")
