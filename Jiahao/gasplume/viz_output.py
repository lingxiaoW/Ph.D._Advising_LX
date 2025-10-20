
"""
输出与作图
- render_frame: 把 2D 浓度切片渲染成一帧 RGB 图
"""

import os
import math
import numpy as np
import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt

import params_randomwalk as P


def render_frame(C: np.ndarray, t: float, npuffs: int,
                 Ux: float, Uy: float,
                 X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    渲染一帧并返回 RGB ndarray。
    - 色标上限按分位数自适应（P.PERC_FOR_VMAX）
    - 左下角画风矢量（Ux, Uy）
    """
    vmax = np.nanpercentile(C, P.PERC_FOR_VMAX)
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1e-6
    vmin = 0.0

    fig, ax = plt.subplots(figsize=(7.5, 4.0), dpi=120)
    im = ax.imshow(
        C, origin="lower",
        extent=[P.Xmin, P.Xmax, P.Ymin, P.Ymax],
        interpolation="nearest",
        vmin=vmin, vmax=vmax
    )
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("concentration (arb. unit)")

    ax.set_title(
        f"Random-walk wind  z={P.z_slice} m | t={int(t)} s | puffs={npuffs} | "
        f"U=({Ux:.2f},{Uy:.2f}) m/s | maxC={C.max():.3e}"
    )
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")

    # 源位置
    ax.plot([P.SRC[0]], [P.SRC[1]], marker="*", markersize=10, color="white")

    # 风矢量
    ax.quiver(P.Xmin + 20, P.Ymin + 20, Ux, Uy,
              angles='xy', scale_units='xy', scale=1, width=0.004)

    fig.tight_layout()

    # 用 renderer.buffer_rgba() 获取像素
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    frame = np.asarray(renderer.buffer_rgba())[:, :, :3]  # RGBA → RGB
    plt.close(fig)
    return frame


def save_gif(frames: list[np.ndarray], filename: str) -> str:
    """把帧列表写成 GIF，并返回输出路径。"""
    out_path = os.path.abspath(filename)
    imageio.mimsave(out_path, frames, duration=0.12)
    return out_path
