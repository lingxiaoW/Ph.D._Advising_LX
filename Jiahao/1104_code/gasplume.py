"""
环境模型代码文件：生成烟雾和吹风
边界：X=[-50,300], Y=[-150,150], Z=[0,20]
原点：SRC=(0,0,1.5)
风速：U_MEAN=2.0 m/s → wind_vec_at(t)=(2.0, 0.0)
时间：TOTAL_T=360s，步长 DT=0.02s
"""
from typing import List, Tuple
import pickle
import numpy as np

DT: float = 0.02               
TOTAL_T: float = 6 * 60.0      # 总时长 [s]

#边界
XMIN, XMAX = -50.0, 300.0
YMIN, YMAX = -150.0, 150.0
ZMIN, ZMAX = 0.0, 20.0

# 烟雾原点 和 wind参数
SRC: Tuple[float, float, float] = (0.0, 0.0, 1.5)
U_MEAN: float = 2.0  # 风 +x [m/s]

EMIT_RATE: float = 5.0  # 释放率 [puffs/s]
Q_PUFF: float = 50.0    # puff权重（用于浓度换算）  

# 横向/竖向扩散参数
A_Y: float = 1.2
A_Z: float = 0.8

# puff 列表：每个元素是 [x, y, z, age]
puffs: List[List[float]] = []


# =====================

def save_puffs(path: str = "puffs_init.pkl") -> None:
    """保存当前 puff 列表，用于测试。"""
    with open(path, "wb") as f:
        pickle.dump(puffs, f)
    print(f"[PLUME] Saved {len(puffs)} puffs -> {path}")


def load_puffs(path: str = "puffs_init.pkl") -> None:
    """加载 puff 列表到全局变量 `puffs`。"""
    global puffs
    with open(path, "rb") as f:
        puffs = pickle.load(f)
    print(f"[PLUME] Loaded {len(puffs)} puffs from {path}")


# 风场


def wind_vec_at(t: float) -> Tuple[float, float]:
    """return(Ux, Uy)。当前为恒定 +x wind模型，与 t 无关。"""
    return (U_MEAN, 0.0)


def _emit_puffs(dt: float) -> None:
    """释放新puff。"""
    n_expected = EMIT_RATE * dt
    n_new = int(n_expected) + (np.random.rand() < (n_expected - int(n_expected)))
    for _ in range(n_new):
        puffs.append([SRC[0], SRC[1], SRC[2], 0.0])


def _advect_and_age(dt: float) -> None:
    """用恒定风做水平平流，并增加 age。"""
    Ux, Uy = U_MEAN, 0.0
    for p in puffs:
        p[0] += Ux * dt
        p[1] += Uy * dt
        # z不随风变化
        p[3] += dt


def _clip_out_of_bounds() -> None:
    """将越界或非正 age 的 puff 丢弃，避免列表无限增长。"""
    keep: List[List[float]] = []
    for x, y, z, age in puffs:
        if age <= 0.0:
            continue
        if (XMIN - 5.0) <= x <= (XMAX + 5.0) and (YMIN - 5.0) <= y <= (YMAX + 5.0) and (ZMIN - 1.0) <= z <= (ZMAX + 1.0):
            keep.append([x, y, z, age])
    puffs.clear(); puffs.extend(keep)


# =====================

def simulate(total_t: float = TOTAL_T, dt: float = DT, save_path: str | None = None) -> None:
    """仅运行 plume（不含无人机），可在结束时保存 puff 场。"""
    print(f"[PLUME] Simulate plume only: total_t={total_t}s, dt={dt}s, U=({U_MEAN},0.0)m/s")
    t = 0.0
    while t < total_t:
        _emit_puffs(dt)
        _advect_and_age(dt)
        _clip_out_of_bounds()
        t += dt
    print(f"[PLUME] Done. puffs={len(puffs)}")
    if save_path:
        save_puffs(save_path)


# =====================
if __name__ == "__main__":
    # 预热 ~355s 并保存到当前目录
    simulate(total_t=355.0, dt=DT, save_path="puffs_init.pkl")
