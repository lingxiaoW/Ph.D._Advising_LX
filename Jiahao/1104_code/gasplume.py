"""
gasplume.py
from:
  https://github.com/Hammerling-Research-Group/FastGaussianPuff
-恒定风场 U = (2.0, 0.0) m/s，始终沿 +X 方向
"""

import math
import pickle
import numpy as np
from typing import List, Tuple


XMIN, XMAX = -50.0, 300.0
YMIN, YMAX = -150.0, 150.0
ZMIN, ZMAX = 0.0, 20.0

SRC = np.array([0.0, 0.0, 1.5])



EMIT_RATE    = 6.0      # puffs per second
Q_PUFF       = 60.0     # "mass" per puff (arbitrary units)
PUFF_MAX_AGE = 240.0    # max puff age [s]
A_Y          = 1.8      # lateral spread coeff (sigma_y ~ A_Y * sqrt(age))
A_Z          = 1.2      # vertical spread coeff (sigma_z ~ A_Z * sqrt(age))

U_MEAN       = 2.0      # m/s along +X

# each puff is [x, y, z, age]
puffs: List[np.ndarray] = []


def wind_vec_at(t: float) -> Tuple[float, float]:
    """
    Constant wind field:
    - U = (2.0, 0.0) m/s
    - Always blowing along +X direction
    - Independent of time t
    """
    Ux = 2.0
    Uy = 0.0
    return (Ux, Uy)


def _deg(ux: float, uy: float) -> float:
    """Convert (Ux, Uy) into angle in degrees (0°=+X, CCW positive)."""
    if ux == 0.0 and uy == 0.0:
        return float("nan")
    return math.degrees(math.atan2(uy, ux))



def _emit_puffs(dt: float):
    """Emit new puffs at the source."""
    n_new = int(EMIT_RATE * dt)
    for _ in range(n_new):
        puffs.append(np.array([SRC[0], SRC[1], SRC[2], 0.0], dtype=float))


def _advect_and_age(dt: float, t: float):
    """Advect all puffs with the wind and increase age."""
    Ux, Uy = wind_vec_at(t)
    for puff in puffs:
        puff[0] += Ux * dt
        puff[1] += Uy * dt
        puff[3] += dt


def _clip_out_of_bounds():
    """Remove puffs that left the domain or exceeded max age."""
    global puffs
    keep = []
    for puff in puffs:
        x, y, z, age = puff
        if (XMIN <= x <= XMAX and
            YMIN <= y <= YMAX and
            ZMIN <= z <= ZMAX and
            age <= PUFF_MAX_AGE):
            keep.append(puff)
    puffs = keep


def simulate(total_time: float = 360.0, dt: float = 0.5):
    """Run plume simulation and populate global `puffs`."""
    t = 0.0
    while t < total_time:
        _emit_puffs(dt)
        _advect_and_age(dt, t)
        _clip_out_of_bounds()
        t += dt
    print(f"[PLUME] Simulated {len(puffs)} puffs at t={t:.1f}s")


# Save / load

def save_puffs(fname: str = "puffs_init.pkl"):
    with open(fname, "wb") as f:
        pickle.dump(puffs, f)
    print(f"[PLUME] Saved {len(puffs)} puffs -> {fname}")


def load_puffs(fname: str = "puffs_init.pkl"):
    global puffs
    with open(fname, "rb") as f:
        puffs = pickle.load(f)
    print(f"[PLUME] Loaded {len(puffs)} puffs from {fname}")



# Self-test

if __name__ == "__main__":
    print("[PLUME] Self-test start.")
    simulate(total_time=360.0, dt=0.5)

    print("[PLUME] Wind samples (t, Ux, Uy, |U|, deg):")
    for ts in [0.0, 60.0, 120.0, 240.0, 360.0]:
        ux, uy = wind_vec_at(ts)
        mag = math.hypot(ux, uy)
        ang = _deg(ux, uy)
        print(f"  t={ts:5.1f}s  U=({ux:+4.2f},{uy:+4.2f})  |U|={mag:4.2f}  phi={ang:6.2f}°")

    save_puffs()
    print("[PLUME] Self-test end.")
