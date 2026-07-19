"""
Closed-Loop Air Conditioner Temperature Control Simulation
------------------------------------------------------------
Room (room thermal response):    G(s) = K / (tau*s + 1)
Controller (PI):                 C(s) = Kp + Ki/s

The room is modeled as a first-order system: the AC unit dumps in
"cooling power" u(t), and the room temperature error T(t) responds
with gain K and time constant tau (how fast the room reacts).

A PI controller closes the loop around a target (setpoint) temperature,
driving the actual temperature to the setpoint while rejecting the
lag introduced by the room's thermal inertia.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# ---------------------------------------------------------------
# Room parameters: G(s) = K / (tau*s + 1)
# ---------------------------------------------------------------
K   = 1.2      # room gain: steady-state temp change (°C) per unit AC power
tau = 150.0    # time constant (sec): how sluggishly the room responds

G_num = [K]
G_den = [tau, 1]

# ---------------------------------------------------------------
# PI Controller: C(s) = Kp + Ki/s = (Kp*s + Ki)/s
# ---------------------------------------------------------------
Kp = 2.0
Ki = 0.015

C_num = [Kp, Ki]
C_den = [1, 0]

# Open-loop L(s) = C(s)*G(s)
L_num = np.polymul(C_num, G_num)
L_den = np.polymul(C_den, G_den)

# Closed-loop with unity feedback: T(s) = L / (1+L)
CL_num = L_num
CL_den = np.polyadd(L_den, np.pad(L_num, (len(L_den) - len(L_num), 0)))

closed_loop = signal.TransferFunction(CL_num, CL_den)
open_loop_plant = signal.TransferFunction(G_num, G_den)  # room alone, no control

# ---------------------------------------------------------------
# Step response: setpoint change from 25°C -> 22°C (a 3°C step down)
# ---------------------------------------------------------------
T_initial = 25.0     # starting room temperature (°C)
T_target  = 22.0     # desired setpoint (°C)
step_size = T_target - T_initial   # -3°C step

t = np.linspace(0, 1500, 3000)  # 25 minutes of simulated time

# Closed-loop (PI-controlled) response
t_cl, y_cl = signal.step(closed_loop, T=t)
room_temp_closed = T_initial + step_size * y_cl

# Open-loop (room only, no feedback) response for comparison
t_ol, y_ol = signal.step(open_loop_plant, T=t)
room_temp_open = T_initial + step_size * y_ol / K  # normalized to reach same target eventually*K

# ---------------------------------------------------------------
# Performance metrics for the closed-loop response
# ---------------------------------------------------------------
final_value = room_temp_closed[-1]
tolerance = 0.02 * abs(step_size)  # 2% settling band

settled_idx = np.where(np.abs(room_temp_closed - final_value) > tolerance)[0]
settling_time = t_cl[settled_idx[-1]] if len(settled_idx) > 0 else 0.0

overshoot = (room_temp_closed.min() - T_target) if step_size < 0 else (room_temp_closed.max() - T_target)
overshoot_pct = abs(overshoot / step_size) * 100

print(f"Setpoint step         : {T_initial} degree celsius -> {T_target} degree celsius")
print(f"Closed-loop final val : {final_value:.3f} degree celsius")
print(f"Settling time (2%)    : {settling_time:.1f} s ({settling_time/60:.1f} min)")
print(f"Overshoot             : {overshoot_pct:.2f}%")

# ---------------------------------------------------------------
# Plotting graph
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 5.5))

ax.plot(t_cl / 60, room_temp_closed, color="#1f77b4", linewidth=2.2,
        label="Closed-loop (PI controlled) room temp")
ax.plot(t_ol / 60, T_initial + (T_target - T_initial) * (1 - np.exp(-t_ol / tau)),
        color="#c50000", linewidth=1.8, linestyle="--",
        label="Open-loop plant only (no controller)")
ax.axhline(T_target, color="green", linestyle=":", linewidth=1.5, label=f"Setpoint = {T_target}°C")
ax.axhline(T_initial, color="gray", linestyle=":", linewidth=1)

ax.set_title("Air Conditioner Room Temperature: Step Response", fontsize=13, fontweight="bold")
ax.set_xlabel("Time (minutes)")
ax.set_ylabel("Room Temperature (°C)")
ax.legend(loc="upper right", fontsize=9)
ax.grid(True, alpha=0.3)

textstr = (f"K = {K}, τ = {tau}s\n"
           f"Kp = {Kp}, Ki = {Ki}\n"
           f"Settling time ≈ {settling_time/60:.1f} min\n"
           f"Overshoot ≈ {overshoot_pct:.1f}%")
ax.text(0.02, 0.25, textstr, transform=ax.transAxes, fontsize=9,
        verticalalignment="top", bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))

plt.tight_layout()
plt.savefig("ac_step_response.png", dpi=150)  # saves next to the script
plt.show() 
