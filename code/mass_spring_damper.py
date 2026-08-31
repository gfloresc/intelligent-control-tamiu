"""
Simple animation: linear mass-spring-damper
   m*q'' + c*q' + k*q = u
State-space form with x1 = q, x2 = q':
   x' = A x + B u,   y = C x + D u
Classroom use: run it and watch the step response.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle

# ---------------- 1. Physical parameters ----------------
m = 1.0     # mass [kg]
c = 0.4     # damping coefficient [N*s/m]
k = 4.0     # spring stiffness [N/m]

# ---------------- 2. State-space model ----------------
# Exactly the matrices from the slide
A = np.array([[0.0, 1.0],
              [-k/m, -c/m]])       # system dynamics
B = np.array([[0.0],
              [1.0/m]])            # input matrix
C = np.array([[1.0, 0.0]])         # we measure position only
D = np.array([[0.0]])              # no direct feedthrough

# Input: unit step (constant force applied to the mass)
def u(t):
    return 1.0

# ---------------- 3. Simulation (Runge-Kutta 4) ----------------
dt = 0.02                   # integration step [s]
T = 20.0                    # total simulation time [s]
t = np.arange(0.0, T, dt)   # time vector

def f(t, x):
    """Vector field x' = A x + B u(t)."""
    return A @ x + B.flatten() * u(t)

x = np.zeros((len(t), 2))   # state history [q, q']
x[0] = [0.0, 0.0]           # initial condition: at rest at the origin

for i in range(len(t) - 1):
    # the four RK4 slopes
    k1 = f(t[i],        x[i])
    k2 = f(t[i] + dt/2, x[i] + dt/2 * k1)
    k3 = f(t[i] + dt/2, x[i] + dt/2 * k2)
    k4 = f(t[i] + dt,   x[i] + dt * k3)
    x[i+1] = x[i] + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

q = x[:, 0]                 # position
y = (C @ x.T).flatten()     # measured output y = C x

# ---------------- 4. Figure: animation (left) + plot (right) ----------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

# --- Left panel: mechanical drawing ---
L0 = 2.0         # spring rest length [m]
W, H = 1.0, 1.0  # block width and height

ax1.set_xlim(-0.5, L0 + W + 2.0)
ax1.set_ylim(-1.5, 1.5)
ax1.set_aspect('equal')
ax1.axis('off')
ax1.set_title('Mass-spring-damper')

# fixed wall at x = 0
ax1.plot([0, 0], [-1.2, 1.2], color='0.3', lw=4)

# objects updated on every frame
spring, = ax1.plot([], [], color='teal', lw=2)         # spring zigzag
damper, = ax1.plot([], [], color='seagreen', lw=3)     # damper rod
block = Rectangle((L0, -H/2), W, H, fc='lightgray', ec='k')
ax1.add_patch(block)
force_arrow = ax1.annotate('', xy=(0, 0), xytext=(0, 0),
                           arrowprops=dict(arrowstyle='->', color='orange', lw=2))

def spring_shape(x0, x1, n=10, amp=0.25):
    """Return the points of a zigzag between x0 and x1 (the spring)."""
    xs = np.linspace(x0, x1, 2*n + 1)
    ys = np.zeros_like(xs)
    ys[1:-1:2] = amp         # upper peaks
    ys[2:-1:2] = -amp        # lower peaks
    return xs, ys + 0.45     # drawn on the upper half

# --- Right panel: time response ---
ax2.set_xlim(0, T)
ax2.set_ylim(min(y) - 0.1, max(y) + 0.1)
ax2.set_xlabel('t [s]')
ax2.set_ylabel('y = q(t) [m]')
ax2.set_title('Measured output (position)')
ax2.grid(alpha=0.3)
ax2.plot(t, y, color='0.8', lw=1)                   # full curve as reference
trace, = ax2.plot([], [], color='darkred', lw=2)    # growing trace
marker, = ax2.plot([], [], 'o', color='darkred')    # current-time marker

# ---------------- 5. Animation function ----------------
skip = 2   # draw 1 out of every 2 samples so it runs smoothly

def update(i):
    j = i * skip
    xq = L0 + q[j]                       # left edge of the block

    block.set_x(xq)                                  # move the block
    spring.set_data(*spring_shape(0, xq))            # stretch the spring
    damper.set_data([0, xq], [-0.45, -0.45])         # damper rod

    # arrow showing the input force u acting on the mass
    force_arrow.set_position((xq + W, 0))
    force_arrow.xy = (xq + W + 0.8*u(t[j]), 0)

    trace.set_data(t[:j], y[:j])         # growing trace
    marker.set_data([t[j]], [y[j]])
    return spring, damper, block, trace, marker

ani = FuncAnimation(fig, update, frames=len(t)//skip,
                    interval=20, blit=False, repeat=True)

plt.tight_layout()
plt.show()

# To save as a GIF (requires pillow), uncomment:
# ani.save('mass_spring_damper.gif', writer='pillow', fps=30)
