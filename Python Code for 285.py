# Math 285 James Scholar Project
# Predator-Prey Numerical Simulation using RK4
# Ananya Gupta - 655886484

import numpy as np
import matplotlib.pyplot as plt

# ── RK4 solver for 2D system ─────────────────────────────────────────────────
def rk4(f1, f2, p0, y0, t0, tf, h=0.005):
    """Solve p' = f1(p,y), y' = f2(p,y) using RK4. Returns t, p, y arrays."""
    t = np.arange(t0, tf + h, h)
    p = np.zeros(len(t))
    y = np.zeros(len(t))
    p[0], y[0] = p0, y0

    for i in range(len(t) - 1):
        pi, yi = p[i], y[i]

        k1p = f1(pi, yi);              k1y = f2(pi, yi)
        k2p = f1(pi+h/2*k1p, yi+h/2*k1y)
        k2y = f2(pi+h/2*k1p, yi+h/2*k1y)
        k3p = f1(pi+h/2*k2p, yi+h/2*k2y)
        k3y = f2(pi+h/2*k2p, yi+h/2*k2y)
        k4p = f1(pi+h*k3p,   yi+h*k3y)
        k4y = f2(pi+h*k3p,   yi+h*k3y)

        p[i+1] = pi + h/6 * (k1p + 2*k2p + 2*k3p + k4p)
        y[i+1] = yi + h/6 * (k1y + 2*k2y + 2*k3y + k4y)

    return t, p, y


# ── System (1.1): Classic Lotka-Volterra ────────────────────────────────────
def sys11(d, a, k, b):
    f1 = lambda p, y: -d*p + a*p*y
    f2 = lambda p, y:  k*y - b*p*y
    return f1, f2


# ── System (1.3): Carrying capacity M + bounded consumption ─────────────────
def sys13(d, a, k, b, M):
    f1 = lambda p, y: -d*p + a*p * y/(1+y)
    f2 = lambda p, y:  k*y*(M - y) - b*p * y/(1+y)
    return f1, f2


import os
os.makedirs("plots", exist_ok=True)

# ════════════════════════════════════════════════════════════════════════════
# FIGURE 1 – System (1.1): phase portrait + time series for varied ICs
# ════════════════════════════════════════════════════════════════════════════
d, a, k, b = 0.5, 0.02, 1.0, 0.04
f1, f2 = sys11(d, a, k, b)
eq_p, eq_y = k/b, d/a  # coexistence equilibrium = (25, 25)

ics = [
    (eq_p*0.5, eq_y*0.5, "Near eq (small)"),
    (eq_p*1.2, eq_y*0.8, "Near eq (offset)"),
    (eq_p*0.2, eq_y*2.0, "Far from eq"),
    (eq_p*3.0, eq_y*0.3, "Large predator"),
]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle(f"System (1.1) | d={d} a={a} k={k} b={b} | eq=({eq_p},{eq_y})", fontweight='bold')

for p0, y0, label in ics:
    t, p, y = rk4(f1, f2, p0, y0, 0, 80)
    ax1.plot(y, p, label=label, linewidth=1.5)
    ax1.plot(y[0], p[0], 'o', markersize=5)
    ax2.plot(t, p, linewidth=1.2, label=f"Pred – {label}")
    ax2.plot(t, y, '--', linewidth=1.2, alpha=0.7)

ax1.plot(eq_y, eq_p, 'k*', markersize=12, label='Equilibrium')
ax1.set(xlabel="Prey (y)", ylabel="Predator (p)", title="Phase Portrait")
ax1.legend(fontsize=8)
ax2.set(xlabel="Time", ylabel="Population", title="Time Series (solid=pred, dashed=prey)")
ax2.legend(fontsize=7)
plt.tight_layout()
plt.savefig("plots/fig1_system11.png", dpi=150)
plt.close()

# ════════════════════════════════════════════════════════════════════════════
# FIGURE 2 – System (1.1): varying k
# ════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
fig.suptitle("Effect of varying k (prey growth rate) – System (1.1)", fontweight='bold')

for ax, k_val in zip(axes, [0.5, 1.0, 2.0]):
    f1, f2 = sys11(d, a, k_val, b)
    t, p, y = rk4(f1, f2, 5, 10, 0, 80)
    ax.plot(t, y, color='green', label='Prey')
    ax.plot(t, p, color='red', label='Predator')
    ax.set(title=f"k = {k_val}", xlabel="Time", ylabel="Population")
    ax.legend()

plt.tight_layout()
plt.savefig("plots/fig2_vary_k.png", dpi=150)
plt.close()

# ════════════════════════════════════════════════════════════════════════════
# FIGURE 3 – System (1.1): varying d
# ════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
fig.suptitle("Effect of varying d (predator death rate) – System (1.1)", fontweight='bold')

for ax, d_val in zip(axes, [0.2, 0.5, 0.9]):
    f1, f2 = sys11(d_val, a, k, b)
    t, p, y = rk4(f1, f2, 5, 10, 0, 80)
    ax.plot(t, y, color='green', label='Prey')
    ax.plot(t, p, color='red', label='Predator')
    ax.set(title=f"d = {d_val}", xlabel="Time", ylabel="Population")
    ax.legend()

plt.tight_layout()
plt.savefig("plots/fig3_vary_d.png", dpi=150)
plt.close()

# ════════════════════════════════════════════════════════════════════════════
# FIGURE 4 – System (1.3): varying M
# ════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle("System (1.3): Effect of carrying capacity M", fontweight='bold')

d13, a13, k13, b13 = 0.5, 2.0, 0.5, 1.0

for col, M in enumerate([5, 10, 50]):
    f1, f2 = sys13(d13, a13, k13, b13, M)
    t, p, y = rk4(f1, f2, 2, 5, 0, 100)
    p = np.clip(p, 0, 1e6)
    y = np.clip(y, 0, 1e6)

    axes[0, col].plot(t, y, color='green', label='Prey')
    axes[0, col].plot(t, p, color='red', label='Predator')
    axes[0, col].set(title=f"M={M} – Time Series", xlabel="Time", ylabel="Population")
    axes[0, col].legend()

    axes[1, col].plot(y, p, color='purple', linewidth=1.5)
    axes[1, col].plot(y[0], p[0], 'go', markersize=8, label='Start')
    axes[1, col].plot(y[-1], p[-1], 'rs', markersize=8, label='End')
    axes[1, col].set(title=f"M={M} – Phase Portrait", xlabel="Prey (y)", ylabel="Predator (p)")
    axes[1, col].legend()

plt.tight_layout()
plt.savefig("plots/fig4_system13_varyM.png", dpi=150)
plt.close()

print("All plots saved to plots/")
