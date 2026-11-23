# lyapunov_design.py: Session 7, designing controllers with Lyapunov functions
# Download: github.com/gfloresc/intelligent-control, folder code/
# Dr. Gerardo Flores, Texas A&M International University
#
# Three worked examples, each with regulation (go to a point) and
# tracking (follow r(t)). Every run simulates the system TWICE:
# with the controller, and without control (u = 0), and plots both.
#
# MARKS. Each slide of Session 7 has a grey box with a mark. Search
# this file for that mark to find its lines:
#   [EX1]  first order:  x' = a x^2 + u
#   [EX2]  mass, cubic spring, quadratic drag, in state space:
#          x1' = x2,  x2' = -x1^3 - x2|x2| + u
#   [EX3]  two states in cascade (backstepping):
#          x1' = x1^2 + x2,  x2' = u
#   [ERR]  tracking: the error coordinates

import numpy as np
import matplotlib.pyplot as plt

# ===================== CLASS EXAMPLES =====================
SCENARIOS = {
    # --- [EX1] first order, the model says a = 1 ---
    'ex1_reg':        dict(ex=1, ref='zero'),               # x -> 0
    'ex1_track':      dict(ex=1, ref='sine'),               # x -> sin t
    'ex1_wrong_a':    dict(ex=1, ref='sine', a=1.5),        # Exercise 1
    'ex1_wrong_a_k8': dict(ex=1, ref='sine', a=1.5, k1=8.0),

    # --- [EX2] mass with cubic spring and quadratic drag ---
    'ex2_reg':        dict(ex=2, ref='zero', law='pd'),     # PD, no model
    'ex2_track':      dict(ex=2, ref='sine', law='cancel'), # x1 -> sin t
    'ex2_track_pd':   dict(ex=2, ref='sine', law='pd'),     # no model: lags
    'ex2_hold_1':     dict(ex=2, ref='one', law='hold'),    # Exercise 2

    # --- [EX3] cascade, backstepping ---
    'ex3_reg':        dict(ex=3, ref='zero', law='bs'),     # x1 -> 0
    'ex3_track':      dict(ex=3, ref='sine', law='bs'),     # x1 -> sin t
    'ex3_lin_small':  dict(ex=3, ref='zero', law='lin', x0=0.5),  # Exercise 3
    'ex3_lin_big':    dict(ex=3, ref='zero', law='lin', x0=1.5),  # escapes

    # --- add your own scenarios below, one per line ---
    'my_test':        dict(ex=3, ref='sine', law='bs', k1=4.0),
}

SCENARIO = 'ex3_track'     # <- change this name

_s = SCENARIOS[SCENARIO]
EX = _s['ex']                       # 1, 2 or 3
REF = _s.get('ref', 'zero')         # 'zero', 'one' or 'sine'
LAW = _s.get('law', 'cancel')       # see each example; 'none' means u = 0
A_REAL = _s.get('a', 1.0)           # [EX1] true coefficient (model: 1)
K1 = _s.get('k1', 2.0)              # gains
K2 = _s.get('k2', 2.0)
X0 = _s.get('x0', 1.5)              # initial x (or x1); x2 starts at 0
T_END, H = 10.0, 0.001


# ---------------- the reference r(t) and its derivatives ----------------
def reference(t):
    if REF == 'sine':
        return np.sin(t), np.cos(t), -np.sin(t)
    if REF == 'one':
        return 1.0, 0.0, 0.0
    return 0.0, 0.0, 0.0


# ---------------- [EX1] x' = a x^2 + u ----------------
def ex1(t, x, law):
    r, rd, _ = reference(t)
    e = x[0] - r                       # [ERR] error coordinate
    if law == 'none':
        u = 0.0
    else:                              # cancel (model a = 1), supply r', damp e
        u = -x[0]**2 + rd - K1*e
    return np.array([A_REAL*x[0]**2 + u]), u


# ---------------- [EX2] x1' = x2,  x2' = -x1^3 - x2|x2| + u ----------------
def ex2(t, x, law):
    x1, x2 = x
    r, rd, rdd = reference(t)
    e1, e2 = x1 - r, x2 - rd           # [ERR] e1 = x1 - r, e2 = x2 - r'
    if law == 'none':
        u = 0.0
    elif law == 'pd':                  # PD on the error, no model at all
        u = -K1*e1 - K2*e2
    elif law == 'hold':                # Exercise 2: hold x1 = 1
        u = r**3 - K2*x2
    else:                              # 'cancel': V = (e2^2 + k1 e1^2)/2
        u = x1**3 + x2*abs(x2) + rdd - K1*e1 - K2*e2
    return np.array([x2, -x1**3 - x2*abs(x2) + u]), u


# ---------------- [EX3] x1' = x1^2 + x2,  x2' = u ----------------
def ex3(t, x, law):
    x1, x2 = x
    r, rd, rdd = reference(t)
    if law == 'none':
        u = 0.0
    elif law == 'lin':                 # designed on the linearization at 0
        u = -4*x1 - 4*x2
    else:                              # 'bs': backstepping
        e1 = x1 - r                    # [ERR] error of x1
        alpha = -x1**2 + rd - K1*e1    # step 1: the x2 we want (virtual control)
        z = x2 - alpha                 # step 2: error of x2 from alpha
        x1dot = x1**2 + x2
        alpha_dot = -2*x1*x1dot + rdd - K1*(x1dot - rd)
        u = alpha_dot - e1 - K2*z      # gives V' = -k1 e1^2 - k2 z^2
    return np.array([x1**2 + x2, u]), u


def simulate(law):
    f = {1: ex1, 2: ex2, 3: ex3}[EX]
    x = np.array([X0]) if EX == 1 else np.array([X0, 0.0])
    out = []
    for k in range(int(T_END/H) + 1):
        t = k*H
        if k % 10 == 0:
            out.append([t, x[0], reference(t)[0], f(t, x, law)[1]])
        k1 = f(t, x, law)[0]; k2 = f(t + H/2, x + H/2*k1, law)[0]
        k3 = f(t + H/2, x + H/2*k2, law)[0]; k4 = f(t + H, x + H*k3, law)[0]
        x = x + H/6*(k1 + 2*k2 + 2*k3 + k4)
        if np.abs(x).max() > 1e4:              # finite escape: stop
            out.append([t, np.nan, reference(t)[0], np.nan])
            break
    return np.array(out).T


def plot(ctrl, free):
    t, y, r, u = ctrl
    fig, axs = plt.subplots(3, 1, figsize=(7, 6.5), sharex=True)
    if REF != 'zero':
        axs[0].plot(t, r, 'k--', lw=1, label='Reference r(t)')
    axs[0].plot(free[0], free[1], color='#C1666B', lw=1.5, label='Without control (u = 0)')
    axs[0].plot(t, y, color='#1F6F8B', lw=2, label='With control')
    axs[0].set_ylim(min(-1.5, np.nanmin(y) - 0.2), max(2.5, np.nanmax(y) + 0.2))
    axs[0].legend(loc='upper right', fontsize=8)
    axs[1].plot(free[0], free[1] - free[2], color='#C1666B', lw=1.5)
    axs[1].plot(t, y - r, color='#1F6F8B', lw=2)
    axs[1].set_ylim(axs[0].get_ylim()[0] - 1, axs[0].get_ylim()[1])
    axs[2].plot(t, u, color='#1F6F8B', lw=2)
    labels = ['x' if EX == 1 else 'x1', 'Error e = x - r' if EX == 1 else 'Error e1 = x1 - r', 'u']
    for ax, lab in zip(axs, labels):
        ax.set_ylabel(lab); ax.grid(alpha=0.3)
    axs[2].set_xlabel('Time [s]'); axs[0].set_title(SCENARIO)
    fig.tight_layout(); fig.savefig(f'{SCENARIO}.png', dpi=150)
    plt.show()


if __name__ == '__main__':
    ctrl, free = simulate(LAW), simulate('none')
    t, y, r, u = ctrl
    e = y - r
    if np.isnan(free[1][-1]):
        print(f'{SCENARIO}: without control the state escapes to infinity'
              f' at t = {free[0][-1]:.2f} s')
    if np.isnan(y[-1]):
        print(f'  with this controller the state ALSO escapes, at t = {t[-1]:.2f} s')
    else:
        print(f'{SCENARIO} with control: final error {e[-1]:+.4f},'
              f' largest |error| after 5 s {np.abs(e[t > 5]).max():.4f},'
              f' max |u| {np.abs(u).max():.2f}')
    plot(ctrl, free)                   # also saves <SCENARIO>.png
