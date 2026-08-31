"""
Lab 1: simple pendulum (nonlinear) - simulation and animation
   x1 = theta, x2 = theta_dot, u = tau

   x1' = x2
   x2' = -(g/l) sin(x1) - b/(m*l^2) x2 + 1/(m*l^2) u

Classroom use: run it, then switch the controller in control() and
watch how the response follows (or fails to follow) the reference.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ---------------- 1. Physical parameters ----------------
m = 0.5      # mass [kg]
l = 0.3      # rod length [m]
b = 0.05     # viscous friction at the joint [N*m*s]
g = 9.81     # gravity [m/s^2]

J = m * l**2  # equivalent inertia, appears in every term

# ---------------- 2. Desired reference ----------------
# Constant set point for the angle [rad]. theta = 0 is hanging down.
theta_d = np.pi / 4       # 45 degrees

def reference(t):
    """Desired angle. Return a constant, or uncomment a time-varying one."""
    return theta_d
    # return 0.5 * np.sin(0.8 * t)                  # sinusoidal tracking
    # return theta_d if t > 2.0 else 0.0            # delayed step

# ---------------- 3. Control law ----------------
# Gains used by the options below
kp, kd = 2.0, 0.2

def control(t, x):
    """
    Input torque u = tau.
    Only ONE option should be active; the rest stay commented out.
    """
    theta, theta_dot = x
    e = reference(t) - theta        # position error
    e_dot = -theta_dot              # error rate (constant reference)

    # --- Option 0 (default): no control, free response ---
    return 0.0

    # --- Option 1: PD control ---
    # return kp * e + kd * e_dot

    # --- Option 2: PD + gravity compensation ---
    # return m * g * l * np.sin(theta) + kp * e + kd * e_dot

    # --- Option 3: feedback linearization (computed torque) ---
    # v = kp * e + kd * e_dot                       # virtual linear input
    # return m * g * l * np.sin(theta) + b * theta_dot + J * v

    # --- Option 4: constant torque, to see the shifted equilibrium ---
    # return 0.3

# ---------------- 4. Dynamics and simulation (Runge-Kutta 4) ----------------
def f(t, x):
    """Vector field of the pendulum."""
    x1, x2 = x
    u = control(t, x)
    dx1 = x2
    dx2 = -(g / l) * np.sin(x1) - (b / J) * x2 + (1.0 / J) * u
    return np.array([dx1, dx2])

dt = 0.01                   # integration step [s]
T = 10.0                    # total simulation time [s]
t = np.arange(0.0, T, dt)   # time vector

x = np.zeros((len(t), 2))   # state history [theta, theta_dot]
x[0] = [np.pi / 3, 0.0]     # initial condition: 60 deg, at rest

u_hist = np.zeros(len(t))   # torque history, for the plot

for i in range(len(t) - 1):
    u_hist[i] = control(t[i], x[i])
    # the four RK4 slopes
    k1 = f(t[i],        x[i])
    k2 = f(t[i] + dt/2, x[i] + dt/2 * k1)
    k3 = f(t[i] + dt/2, x[i] + dt/2 * k2)
    k4 = f(t[i] + dt,   x[i] + dt * k3)
    x[i+1] = x[i] + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

u_hist[-1] = control(t[-1], x[-1])

theta = x[:, 0]                            # angle
ref = np.array([reference(ti) for ti in t])  # reference over time

# ---------------- 5. Figure: animation (left) + plots (right) ----------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

# --- Left panel: the pendulum ---
L_view = 0.5   # fixed view scale [m], independent of l

ax1.set_xlim(-1.5*L_view, 1.5*L_view)
ax1.set_ylim(-1.5*L_view, 1.0*L_view)
ax1.set_aspect('equal')
ax1.axis('off')
ax1.set_title('Simple pendulum')

# pivot at the origin
ax1.plot(0, 0, 'ko', ms=8)

# dashed line showing the desired angle
ax1.plot([0, l*np.sin(theta_d)], [0, -l*np.cos(theta_d)],
         '--', color='0.6', lw=1.5, label='reference')
ax1.legend(loc='upper right', fontsize=8)

rod, = ax1.plot([], [], color='teal', lw=3)              # rod
bob, = ax1.plot([], [], 'o', color='darkred', ms=16)     # mass at the tip
clock = ax1.text(-1.4*l, 0.85*l, '', fontsize=9)         # time readout

# --- Right panel: angle and torque ---
ax2.set_xlim(0, T)
ax2.set_xlabel('t [s]')
ax2.set_ylabel('theta [rad]')
ax2.set_title('Angle and control torque')
ax2.grid(alpha=0.3)
ax2.plot(t, ref, '--', color='0.6', lw=1.5)   # reference
ax2.plot(t, theta, color='0.85', lw=1)        # full response as reference
trace, = ax2.plot([], [], color='darkred', lw=2)   # growing trace
marker, = ax2.plot([], [], 'o', color='darkred')   # current-time marker

ax3 = ax2.twinx()                              # second y-axis for the torque
ax3.set_ylabel('u [N*m]', color='orange')
ax3.tick_params(axis='y', labelcolor='orange')
torque, = ax3.plot([], [], color='orange', lw=1.5, alpha=0.8)
ax3.set_ylim(min(u_hist) - 0.1, max(u_hist) + 0.1)

# ---------------- 6. Animation function ----------------
skip = 3   # draw 1 out of every 3 samples so it runs smoothly

def update(i):
    j = i * skip
    # tip position from the angle (theta = 0 points down)
    xt = l * np.sin(theta[j])
    yt = -l * np.cos(theta[j])

    rod.set_data([0, xt], [0, yt])       # move the rod
    bob.set_data([xt], [yt])             # move the mass
    clock.set_text(f't = {t[j]:.1f} s')

    trace.set_data(t[:j], theta[:j])     # growing trace
    marker.set_data([t[j]], [theta[j]])
    torque.set_data(t[:j], u_hist[:j])   # applied torque
    return rod, bob, trace, marker, torque, clock

ani = FuncAnimation(fig, update, frames=len(t)//skip,
                    interval=20, blit=False, repeat=True)

plt.tight_layout()
plt.show()

# To save as a GIF (requires pillow), uncomment:
# ani.save('pendulum.gif', writer='pillow', fps=30)
