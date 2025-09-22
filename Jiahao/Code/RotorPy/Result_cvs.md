# Result: `uav_telemetry.csv` 

## Columns (in order)

1. **`time`** — Simulation time in **seconds (s)**  
   - Monotonic; starts near `0` and ends at `t_final` (e.g., `10.0`).

2. **`x`, `y`, `z`** — Vehicle position in **meters (m)**
   - (up is positive in most RotorPy setups).

3. **`vx`, `vy`, `vz`** — Linear velocity components in **m/s**  
   - Speed magnitude:
     $$\text{speed} = \sqrt{vx^2 + vy^2 + vz^2}$$

4. **`yaw_deg`, `pitch_deg`, `roll_deg`** — Euler angles (ZYX order) in **degrees (deg)**  
   - `yaw_deg`: heading about **Z** (0° aligned with +X, positive CCW toward +Y)  
   - `pitch_deg`: rotation about intermediate **Y**  
   - `roll_deg`: rotation about intermediate **X**  
   - If the simulation results **lack quaternions**, these three columns are filled with `NaN`.

---

## Sampling & File Size

- Written approximately at the simulation rate **`sim_rate`** (default **100 Hz**).  
- With `t_final = 10 s`, expect ~**1000 rows** (+ 1 header row).


