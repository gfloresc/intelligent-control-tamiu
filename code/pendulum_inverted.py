# SENG 5360 - Intelligent Control
# Session 4, Code 4.1: animated upright pendulum
# Dr. Gerardo Flores, Texas A&M International University
#
# Same pendulum and gains as Sessions 3 and 4. Pick a class
# example below, run, and watch what the controller does.
#
# The controller runs like a real board: every TS seconds it
# reads the angle, estimates the velocity, computes the torque
# and holds it until the next reading.
#
# MARKS. Each slide of Session 4 has a box "In Code 4.1" with a
# mark. Search this file for that mark to find its lines:
#   [Q1]    Question 1: the real mass is not the design mass
#   [Q2]    Question 2: a constant push
#   [FIX-I] the practical fix: integral action
#   [Q3]    Question 3: a noisy sensor
#   [FIX-F] the practical fix: a low-pass filter

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ===================== CLASS EXAMPLES =====================
# Choose one scenario by name in SCENARIO below and run again.
# Each line says which slide it reproduces and what you should
# see. Anything a line does not set keeps the Session 3 value.
SCENARIOS = {
    # --- [Q1] Question 1: what if the model is wrong? ---
    'design':      dict(m=0.5),          # back up in about 1 s
    'heavy':       dict(m=1.0),          # back up, but slowly
    'too_heavy':   dict(m=1.1),          # stays ~0.52 rad below

    # --- [Q1] Exercise 1: the rule k1 > m g l ---
    # (1) largest mass k1/(g l) = 1.05 kg: compare 'heavy'
    #     and 'too_heavy', one on each side of it
    # (2) a 2 kg load needs k1 > 5.886; we choose k1 = 8
    'load_2kg':    dict(m=2.0),          # falls, k1 too small
    'load_2kg_k8': dict(m=2.0, k1=8.0),  # works again

    # --- [Q2] Question 2 and Exercise 2: a constant push ---
    'push_0.1':    dict(d=0.1),          # offset 0.062 rad
    'push_0.2':    dict(d=0.2),          # offset 0.123 rad
    'push_high_k': dict(d=0.2, k1=11.5), # offset 0.02 rad

    # --- [FIX-I] The practical fix: integral action ---
    'integral':    dict(d=0.2, ki=3.0),  # offset goes to zero
                                         # and u settles at -0.2:
                                         # it cancels the push
    'integral_too_much': dict(d=0.2, ki=15.0),  # swings grow

    # --- [Q3] Question 3 and Exercise 3: a noisy sensor ---
    'noise_raw':   dict(sigma=0.01),     # torque jitters a lot
    'noise_fast':  dict(sigma=0.01, ts=0.001),  # 10x faster:
                                         # jitter gets WORSE

    # --- [FIX-F] The practical fix: a low-pass filter ---
    'noise_filtered':    dict(sigma=0.01, tau=0.05),  # smooth
    'noise_slow_filter': dict(sigma=0.01, tau=0.5),   # late:
                                         # overshoots, keeps swinging
}

SCENARIO = 'design'     # <- change this name ['design' by default]

_s = SCENARIOS[SCENARIO]
M_REAL = _s.get('m', 0.5)      # real mass [kg]
K1 = _s.get('k1', 3.0915)      # proportional gain
D_PUSH = _s.get('d', 0.0)      # constant push [N.m], from t = 1 s
KI = _s.get('ki', 0.0)         # integral gain (0 = plain PD)
SIGMA = _s.get('sigma', 0.0)   # encoder noise [rad] (0 = perfect)
TAU = _s.get('tau', None)      # filter constant [s] (None = off)
TS = _s.get('ts', 0.01)        # sample time of the controller [s]

# ---------------- fixed data ----------------
G, L, B = 9.81, 0.3, 0.05      # gravity, length, friction
K2 = 0.3280                    # derivative gain, never changed
THETA0 = np.pi - 0.3           # start 0.3 rad away from upright
T_END, FPS = 8.0, 30           # simulated time and frame rate
H = 0.001                      # integration step of the plant [s]


# ---------------- the pendulum ----------------
def push(t):
    """[Q2] Constant disturbance torque d(t), on from t = 1 s."""
    return D_PUSH if t >= 1.0 else 0.0


def plant(x, u, t):
    """True pendulum: x = [theta, omega], theta from straight down."""
    theta, omega = x
    # m l^2 theta'' = -m g l sin(theta) - b theta' + u + d
    # [Q1] M_REAL is the REAL mass; the gains never see it
    # [Q2] the push d enters HERE, added to the motor torque u
    domega = (-M_REAL*G*L*np.sin(theta) - B*omega
              + u + push(t)) / (M_REAL*L**2)
    return np.array([omega, domega])


# ---------------- the sensor ----------------
def sensor(theta, rng):
    """Encoder: the true angle plus random noise of size SIGMA."""
    # [Q3] the noise enters HERE:  y_k = theta_k + eta_k
    return theta + SIGMA*rng.standard_normal()


# ---------------- the controller ----------------
def velocity(y, mem):
    """Velocity from two readings, then the low-pass filter."""
    # [Q3] w_k = (y_k - y_{k-1}) / Ts: dividing by Ts amplifies noise
    w = (y - mem['y_old']) / TS
    mem['y_old'] = y
    if TAU is None:
        return w                         # filter off
    # [FIX-F] w_f_k = w_f_{k-1} + (Ts/tau) (w_k - w_f_{k-1})
    mem['w_f'] += (TS/TAU) * (w - mem['w_f'])
    return mem['w_f']                    # filtered velocity


def controller(y, mem):
    """PD of Session 3, plus integral action when KI > 0."""
    e = y - np.pi                        # angle error
    w = velocity(y, mem)
    # [FIX-I] z_k = z_{k-1} + Ts e_k: the accumulated error
    mem['z'] += e*TS
    # [FIX-I] u = -k1 e - k2 w - ki z   (ki = 0: the PD of Session 3)
    return -K1*e - K2*w - KI*mem['z']


# ---------------- the simulation ----------------
def simulate():
    """Run plant and controller; keep one sample per video frame."""
    rng = np.random.default_rng(1)
    x = np.array([THETA0, 0.0])
    mem = {'y_old': THETA0, 'w_f': 0.0, 'z': 0.0}
    every = round(TS/H)                  # plant steps per reading
    keep = round(1/(FPS*H))              # plant steps per frame
    t_out, th_out, u_out = [], [], []
    u = 0.0
    for k in range(int(T_END/H) + 1):
        t = k*H
        if k % every == 0:               # a new reading arrives
            # [Q3] the controller gets the reading y, never x[0]
            u = controller(sensor(x[0], rng), mem)
        if k % keep == 0:
            t_out.append(t); th_out.append(x[0]); u_out.append(u)
        # one Runge-Kutta step of the true pendulum
        k1 = plant(x, u, t)
        k2 = plant(x + H/2*k1, u, t + H/2)
        k3 = plant(x + H/2*k2, u, t + H/2)
        k4 = plant(x + H*k3, u, t + H)
        x = x + H/6*(k1 + 2*k2 + 2*k3 + k4)
    return np.array(t_out), np.array(th_out), np.array(u_out)


# ---------------- the animation ----------------
def animate(t, theta, u):
    """Pendulum on the left; angle error and torque on the right."""
    fig = plt.figure(figsize=(10, 4.5))
    ax0 = fig.add_subplot(1, 2, 1)
    ax1 = fig.add_subplot(2, 2, 2)
    ax2 = fig.add_subplot(2, 2, 4, sharex=ax1)

    ax0.set_xlim(-0.4, 0.4); ax0.set_ylim(-0.4, 0.4)
    ax0.set_aspect('equal'); ax0.axis('off')
    ax0.plot([0, 0], [0, L], 'k--', lw=1)           # upright goal
    rod, = ax0.plot([], [], lw=3, color='#1F6F8B')
    bob, = ax0.plot([], [], 'o', ms=14, color='#1F6F8B')
    ax0.plot(0, 0, 'ks', ms=6)                       # pivot
    label = ax0.text(-0.38, -0.36, '', fontsize=10)
    ax0.set_title(f'{SCENARIO}', fontsize=11)

    e = theta - np.pi
    ax1.set_xlim(0, T_END)
    ax1.set_ylim(min(-0.4, e.min() - 0.1), max(0.4, e.max() + 0.1))
    ax1.axhline(0, color='k', ls=':', lw=1)
    ax1.set_ylabel(r'$\theta-\pi$ [rad]')
    ax2.set_ylim(min(-1.0, u.min() - 0.2), max(1.0, u.max() + 0.2))
    ax2.axhline(0, color='k', ls=':', lw=1)
    ax2.set_xlabel('t [s]'); ax2.set_ylabel('u [N.m]')
    c1, = ax1.plot([], [], color='#1F6F8B', lw=2)
    c2, = ax2.plot([], [], color='#C1666B', lw=1)
    fig.tight_layout()

    def update(k):
        # theta is measured from the downward vertical
        x, y = L*np.sin(theta[k]), -L*np.cos(theta[k])
        rod.set_data([0, x], [0, y]); bob.set_data([x], [y])
        c1.set_data(t[:k+1], e[:k+1]); c2.set_data(t[:k+1], u[:k+1])
        wind = '   push ON' if push(t[k]) else ''
        label.set_text(f't = {t[k]:.1f} s{wind}')
        return rod, bob, c1, c2, label

    return FuncAnimation(fig, update, frames=len(t),
                         interval=1000/FPS, blit=True)


def show(anim):
    """Inline player in Colab or Jupyter, a window otherwise."""
    try:
        from IPython import get_ipython
        from IPython.display import HTML, display
        if get_ipython() is not None:
            plt.close()
            display(HTML(anim.to_jshtml()))
            return
    except ImportError:
        pass
    plt.show()


if __name__ == '__main__':
    t, theta, u = simulate()
    print(f'{SCENARIO}: final error {theta[-1] - np.pi:+.3f} rad,'
          f' torque jitter after 2 s {u[t > 2].std():.2f} N.m')
    show(animate(t, theta, u))