# pendulum_tracking.py: Session 5, tracking with the actuated pendulum
# Download: github.com/gfloresc/intelligent-control, folder code/
# Dr. Gerardo Flores, Texas A&M International University
#
# Same pendulum as Sessions 1 to 4 (m = 0.5 kg, l = 0.3 m, b = 0.05,
# motor torque u at the pivot). Now the goal is to FOLLOW a path
# theta_d(t), not only to hold a point. Pick a class example below.
#
# MARKS. The slides have grey boxes with a mark. Search for it:
#   [PATH]   the desired path theta_d(t) and its derivatives
#   [ERR]    the tracking error e = theta - theta_d
#   [CT]     the computed-torque controller (cancel, then PD)
#   [PD]     PD on the error with acceleration feedforward, for comparison
#   [MODEL]  the model the controller believes (may be wrong)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ===================== CLASS EXAMPLES =====================
SCENARIOS = {
    # --- [PATH] regulation versus tracking ---
    'hold_top':      dict(path='hold', target=np.pi),   # regulation
    'hold_side':     dict(path='hold', target=np.pi/2), # hold horizontal
    'sine':          dict(path='sine'),                 # swing around bottom

    # --- Exercise 1: the cheapest swing (A = 0.2 rad) ---
    # start='path': the pendulum starts ON the path, so the printed torque
    # is the torque of the path itself, with no start-up transient
    'swing_slow':    dict(path='sine', amp=0.2, freq=2.0, start='path'),
    'swing_natural': dict(path='sine', amp=0.2, freq=5.72, start='path'),
    'swing_fast10':  dict(path='sine', amp=0.2, freq=10.0, start='path'),
    'swing_top':     dict(path='sine', amp=0.2, freq=5.72, center=np.pi,
                          start='path'),

    # --- Exercise 2: spin at constant speed, with or without friction comp. ---
    'spin':          dict(path='spin'),                 # error goes to 0
    'spin_no_fc':    dict(path='spin', fc=False),       # constant lag

    # --- [CT] versus [PD] on a swing-up path ---
    'swing_up':      dict(path='swing'),                # computed torque
    'swing_up_pd':   dict(path='swing', law='pd'),      # PD only: stuck
    'swing_up_pd_high': dict(path='swing', law='pd', kp=64.0, kd=16.0),  # up,
                                                        # large error on the way

    # --- wrong model: the cancellation is imperfect ---
    'heavy':         dict(path='hold', target=np.pi/2, m=0.6),  # offset
    'heavy_high_k':  dict(path='hold', target=np.pi/2, m=0.6, kp=64.0, kd=16.0),
    'heavy_integral': dict(path='hold', target=np.pi/2, m=0.6, ki=20.0),
    'heavy_swing':   dict(path='swing', m=0.6),         # tracks with error

    # --- motor limit (Session 3) ---
    'swing_limited': dict(path='swing', umax=1.2),      # saturates, fails
    'swing_fast':    dict(path='swing', T=1.0),         # needs more torque

    # --- Exercise 3: a straight-line path instead of the half cosine ---
    'ramp_up':       dict(path='ramp'),                 # falls behind, overshoots

    # --- add your own scenarios below ---
    'my_test':       dict(path='swing', T=2.0, m=0.55),
}

SCENARIO = 'swing_up'     # <- change this name

_s = SCENARIOS[SCENARIO]
PATH = _s.get('path', 'swing')      # 'hold', 'sine' or 'swing'
TARGET = _s.get('target', np.pi)    # angle to hold, for 'hold'
T_PATH = _s.get('T', 3.0)           # duration of the swing-up [s]
LAW = _s.get('law', 'ct')           # 'ct' computed torque, 'pd' PD + feedforward
FRIC_COMP = _s.get('fc', True)      # cancel friction in 'ct'?
AMP = _s.get('amp', 0.8)            # 'sine': amplitude [rad]
FREQ = _s.get('freq', 2.0)          # 'sine': frequency [rad/s]
CENTER = _s.get('center', 0.0)      # 'sine': center angle [rad]
OMEGA = _s.get('omega', 2.0)        # 'spin': constant speed [rad/s]
M_REAL = _s.get('m', 0.5)           # real mass [kg]
KP = _s.get('kp', 16.0)             # error dynamics: e'' + KD e' + KP e = 0
KD = _s.get('kd', 8.0)
KI = _s.get('ki', 0.0)              # integral gain on e (Session 4)
U_MAX = _s.get('umax', 3.0)         # motor limit [N.m]
START = _s.get('start', 'rest')     # 'rest': hanging at rest; 'path': on the path

# ---------------- fixed data ----------------
L, G, B = 0.3, 9.81, 0.05
M_MODEL = 0.5                       # [MODEL] mass the controller believes
T_END, H, FPS = 6.0, 0.001, 30


# ---------------- [PATH] the desired path ----------------
def path(t):
    """Return theta_d, theta_d', theta_d'' at time t."""
    if PATH == 'hold':
        return TARGET, 0.0, 0.0
    if PATH == 'sine':                      # CENTER + AMP sin(FREQ t)
        a, w = AMP, FREQ
        return (CENTER + a*np.sin(w*t), a*w*np.cos(w*t),
                -a*w**2*np.sin(w*t))
    if PATH == 'spin':                      # turn forever at speed OMEGA
        return OMEGA*t, OMEGA, 0.0
    if PATH == 'ramp':                      # straight line from 0 to pi in T
        if t >= T_PATH:
            return np.pi, 0.0, 0.0
        return np.pi*t/T_PATH, np.pi/T_PATH, 0.0
    # 'swing': smooth rise from 0 to pi in T_PATH seconds, then hold
    if t >= T_PATH:
        return np.pi, 0.0, 0.0
    w = np.pi/T_PATH
    return (np.pi/2*(1 - np.cos(w*t)),
            np.pi/2*w*np.sin(w*t),
            np.pi/2*w**2*np.cos(w*t))


# ---------------- the true pendulum ----------------
def plant(x, u):
    th, om = x
    dom = (-M_REAL*G*L*np.sin(th) - B*om + u) / (M_REAL*L**2)
    return np.array([om, dom])


# ---------------- the controllers ----------------
def controller(t, x, mem):
    th, om = x
    thd, thd_dot, thd_ddot = path(t)
    e, e_dot = th - thd, om - thd_dot               # [ERR]
    mem['z'] += e*H                                 # integral of e
    v = thd_ddot - KD*e_dot - KP*e - KI*mem['z']    # wanted e''
    if LAW == 'pd':                                 # [PD] no gravity/friction comp.
        return M_MODEL*L**2 * v
    # [CT] computed torque: cancel gravity and friction, then PD
    friction = B*om if FRIC_COMP else 0.0
    return M_MODEL*L**2 * v + M_MODEL*G*L*np.sin(th) + friction


def simulate():
    if START == 'path':                 # on the path: theta(0), theta'(0)
        x = np.array(path(0.0)[:2], dtype=float)
    else:                               # hanging down, at rest
        x = np.array([0.0, 0.0])
    mem = {'z': 0.0}
    keep = round(1/(FPS*H))
    out = []
    for k in range(int(T_END/H) + 1):
        t = k*H
        u = float(np.clip(controller(t, x, mem), -U_MAX, U_MAX))
        if k % keep == 0:
            out.append([t, x[0], path(t)[0], u])
        k1 = plant(x, u); k2 = plant(x + H/2*k1, u)
        k3 = plant(x + H/2*k2, u); k4 = plant(x + H*k3, u)
        x = x + H/6*(k1 + 2*k2 + 2*k3 + k4)
    return np.array(out).T


# ---------------- animation ----------------
def animate(t, th, thd, u):
    fig = plt.figure(figsize=(10, 4.5))
    ax0 = fig.add_subplot(1, 2, 1)
    ax1 = fig.add_subplot(3, 2, 2); ax2 = fig.add_subplot(3, 2, 4)
    ax3 = fig.add_subplot(3, 2, 6)
    ax0.set_xlim(-0.4, 0.4); ax0.set_ylim(-0.4, 0.4)
    ax0.set_aspect('equal'); ax0.axis('off'); ax0.set_title(SCENARIO)
    ghost, = ax0.plot([], [], lw=2, ls='--', color='#CCCCCC')   # desired
    rod, = ax0.plot([], [], lw=3, color='#1F6F8B')
    bob, = ax0.plot([], [], 'o', ms=14, color='#1F6F8B')
    ax0.plot(0, 0, 'ks', ms=6)
    label = ax0.text(-0.38, -0.36, '', fontsize=10)
    e = th - thd
    for ax, y, lab in [(ax1, (thd, th), r'$\theta$ [rad]'),
                       (ax2, (e,), r'$e=\theta-\theta_d$'),
                       (ax3, (u,), 'u [N.m]')]:
        ax.set_xlim(0, T_END); ax.axhline(0, color='k', ls=':', lw=1)
        lo, hi = min(v.min() for v in y), max(v.max() for v in y)
        pad = 0.1*(hi - lo) + 0.05
        ax.set_ylim(lo - pad, hi + pad); ax.set_ylabel(lab)
    ax3.set_xlabel('t [s]')
    c_d, = ax1.plot([], [], ls='--', color='#55606E', label='desired')
    c_th, = ax1.plot([], [], color='#1F6F8B', label='actual')
    ax1.legend(loc='lower right', fontsize=8)
    c_e, = ax2.plot([], [], color='#C1666B')
    c_u, = ax3.plot([], [], color='#4C956C')
    fig.tight_layout()

    def update(k):
        for line, ang in [(rod, th[k]), (ghost, thd[k])]:
            line.set_data([0, L*np.sin(ang)], [0, -L*np.cos(ang)])
        bob.set_data([L*np.sin(th[k])], [-L*np.cos(th[k])])
        c_d.set_data(t[:k+1], thd[:k+1]); c_th.set_data(t[:k+1], th[:k+1])
        c_e.set_data(t[:k+1], e[:k+1]); c_u.set_data(t[:k+1], u[:k+1])
        label.set_text(f't = {t[k]:.1f} s')
        return ghost, rod, bob, c_d, c_th, c_e, c_u, label

    return FuncAnimation(fig, update, frames=len(t),
                         interval=1000/FPS, blit=True)


def save_summary(t, th, thd, u):
    """Save angle, error and torque to <SCENARIO>.png, for the homework."""
    fig, axs = plt.subplots(3, 1, figsize=(6, 5), sharex=True)
    axs[0].plot(t, thd, 'k--', lw=1, label='Desired')
    axs[0].plot(t, th, color='#1F6F8B', label='Actual')
    axs[0].legend(loc='best', fontsize=8)
    axs[1].plot(t, th - thd, color='#C1666B')
    axs[2].plot(t, u, color='#4C956C')
    for ax, lab in zip(axs, [r'$\theta$ [rad]', r'$e$ [rad]', r'$u$ [N.m]']):
        ax.set_ylabel(lab); ax.grid(alpha=0.3)
    axs[2].set_xlabel('Time [s]'); axs[0].set_title(SCENARIO)
    fig.tight_layout(); fig.savefig(f'{SCENARIO}.png', dpi=150)
    plt.close(fig)


def show(anim):
    """Inline player in Colab or Jupyter, a window otherwise."""
    try:
        from IPython import get_ipython
        from IPython.display import HTML, display
        if get_ipython() is not None:
            plt.close(); display(HTML(anim.to_jshtml())); return
    except ImportError:
        pass
    plt.show()


if __name__ == '__main__':
    t, th, thd, u = simulate()
    e = th - thd
    print(f'{SCENARIO}: max |e| {np.abs(e).max():.3f} rad,'
          f' final e {e[-1]:+.4f} rad, max |u| {np.abs(u).max():.3f} N.m')
    save_summary(t, th, thd, u)          # writes <SCENARIO>.png
    show(animate(t, th, thd, u))
