"""
Sensor Reading
在无人机位置获取气体浓度与wind信息。
"""
from __future__ import annotations
import math
from typing import Iterable, Sequence, Tuple, Optional

# puffs/参数/风场
import gasplume as plume

# 数值保护常量
_EPS_SIG: float = 1e-6                 # 标准差下限，避免除零
_PI2_3: float = (2.0 * math.pi) ** 1.5 # 常数 (2π)^{3/2}

__all__ = ["sensor_reading",]


def _puff_contrib(
    x: float, y: float, z: float,
    puff: Sequence[float],
    *, q_puff: float, a_y: float, a_z: float,) -> float:
    """
    计算单个** puff 在点 (x,y,z) 处的浓度贡献。
    """
    xp, yp, zp, age = puff
    if age <= 0.0:
        return 0.0

    sig_y = max(_EPS_SIG, a_y * math.sqrt(age))
    sig_z = max(_EPS_SIG, a_z * math.sqrt(age))

    norm = q_puff / (_PI2_3 * (sig_y ** 2) * sig_z)

    dx, dy = (x - xp), (y - yp)
    horiz = math.exp(-0.5 * (dx * dx + dy * dy) / (sig_y ** 2))

    dz = z - zp
    vz = math.exp(-0.5 * (dz * dz) / (sig_z ** 2))
    vz_img = math.exp(-0.5 * ((z + zp) ** 2) / (sig_z ** 2))

    return norm * horiz * (vz + vz_img)


def _sum_concentration(
    x: float, y: float, z: float,
    puffs: Iterable[Sequence[float]],
    *, q_puff: float, a_y: float, a_z: float,) -> float:
    """累加一组 puff 的浓度贡献"""
    total = 0.0
    for puff in puffs:
        total += _puff_contrib(x, y, z, puff, q_puff=q_puff, a_y=a_y, a_z=a_z)
    return total


def _resolve_q_puff() -> float:
    """从 plume 模块解析 PUFF_Q"""
    if hasattr(plume, "PUFF_Q"):
        return float(getattr(plume, "PUFF_Q"))
    if hasattr(plume, "Q_PUFF"):
        return float(getattr(plume, "Q_PUFF"))
    raise AttributeError("Neither PUFF_Q nor Q_PUFF found in gasplume_xy_final.py")

###### Input: (x, y, z) 
def sensor_reading(
    x: float,
    y: float,
    z: float,
    t: float,
    *,
    puffs_override: Optional[Iterable[Sequence[float]]] = None,
    degrees: bool = True,) -> Tuple[float, float, float]:

    # 风场
    #   wind_vec_at(t) → 返回风在水平面的分量 (Ux, Uy)
    # 
    #   Ux: 风在 +X 方向的速度分量 [m/s]
    #   Uy: 风在 +Y 方向的速度分量 [m/s]
    #   wind_v = sqrt(Ux^2 + Uy^2)
    Ux, Uy = plume.wind_vec_at(t)
    v = math.hypot(Ux, Uy)

    #       wind_phi = atan2(Uy, Ux)
    #       (再由弧度转换为角度: degrees(wind_phi))
    #       0°   → 风朝 +X 方向
    #       90°  → 风朝 +Y 方向
    #      -90°  → 风朝 -Y 方向
    #      ±180° → 风朝 -X 方向
    phi = math.degrees(math.atan2(Uy, Ux)) if degrees else math.atan2(Uy, Ux)

    # 浓度
    q_puff = _resolve_q_puff()
    a_y = float(getattr(plume, "A_Y"))
    a_z = float(getattr(plume, "A_Z"))

    # 获取最新 puff 列表
    puffs_src = puffs_override if puffs_override is not None else getattr(plume, "puffs")
    c = _sum_concentration(x, y, z, puffs_src, q_puff=q_puff, a_y=a_y, a_z=a_z)
    
    ###  Output: gas_concentration (c), wind_speed (v), wind_direction (phi).
    return c, v, phi


if __name__ == "__main__":
    # test（仅演示调用）：
    print("[SensorReading] 运行期绑定 plume 全局变量（puffs/参数/风场）。")
    xq, yq, zq, t_demo = 5.0, 0.0, 1.5, 10.0
    c, v, phi = sensor_reading(xq, yq, zq, t_demo, degrees=True)
    print(f"pos=({xq},{yq},{zq}), t={t_demo}s -> c={c:.6e}, |U|={v:.2f} m/s, dir={phi:.1f}°")
