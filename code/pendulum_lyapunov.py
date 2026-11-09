# pendulum_lyapunov.py: Session 6, energy and Lyapunov functions
# Download: github.com/gfloresc/intelligent-control, folder code/
# Dr. Gerardo Flores, Texas A&M International University
#
# Same pendulum as Sessions 1 to 5 (m = 0.5 kg, l = 0.3 m, b = 0.05).
# Pick a class example below, run, and watch the pendulum together
# with its energy (or Lyapunov function) V(t).
#
# MARKS. Each slide of Session 6 has a grey box with a mark. Search
# this file for that mark to find its lines:
#   [ENERGY] the energy of the pendulum and its rate of change
#   [PDG]    Design 1: PD + gravity compensation, from a Lyapunov function
#   [SWING]  Design 2: energy pumping with a weak motor
#   [CATCH]  switching to the balance controller near the top

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ===================== CLASS EXAMPLES =====================
SCENARIOS = {
    # --- [ENERGY] the free pendulum (no motor) ---
    'free':          dict(law='none', theta0=2.5),          # E decreases
    'free_no_fric':  dict(law='none', theta0=2.5, b=0.0),   # E constant
    'free_top':      dict(law='none', theta0=np.pi),        # stays up?

    # --- [PDG] Design 1: hold any angle ---
    'hold_side':     dict(law='pdg', target=np.pi/2),       # horizontal
    'hold_120':      dict(law='pdg', target=2*np.pi/3),     # 120 degrees
    'hold_top':      dict(law='pdg', target=np.pi),         # upright
    'hold_wrong_m':  dict(law='pdg', target=np.pi/2, m=0.6),  # offset

    # --- [SWING] Design 2: swing up with a 1.2 N.m motor ---
    'swing_up':      dict(law='swing'),                      # pump, catch
    'swing_no_catch': dict(law='swing', catch=False),        # creeps to the top
    'swing_soft':    dict(law='swing', k=2.0),               # gentle pump
    'swing_weaker':  dict(law='swing', umax=0.6),            # 0.6 N.m
    'swing_at_rest': dict(law='swing', theta0=0.0),          # never starts
    'swing_too_weak': dict(law='swing', umax=0.35),          # never arrives

    # --- Lab 6: add your own scenarios below, one per line ---
    'my_test':       dict(law='pdg', target=np.pi/4),
}

SCENARIO = 'swing_up'     # <- change this name

_s = SCENARIOS[SCENARIO]
LAW = _s.get('law', 'swing')        # 'none', 'pdg' or 'swing'
M_REAL = _s.get('m', 0.5)           # real mass [kg]
M_MODEL = 0.5                       # mass the controller believes [kg]
B = _s.get('b', 0.05)               # friction
TARGET = _s.get('target', np.pi)    # 'pdg': angle to hold [rad]
KP, KD = _s.get('kp', 16.0), _s.get('kd', 8.0)   # 'pdg' and catch gains
K_PUMP = _s.get('k', 20.0)          # 'swing': pumping gain
U_MAX = _s.get('umax', 1.2)         # motor limit [N.m] (weaker than mgl!)
CATCH = _s.get('catch', True)       # 'swing': switch to balance near top
THETA0 = _s.get('theta0', 0.05)     # initial angle [rad], at rest

G, L = 9.81, 0.3
T_END, H, FPS = 15.0, 0.001, 30


# ---------------- [ENERGY] the pendulum and its energy ----------------
def plant(x, u):
    """m l^2 theta'' = -m g l sin(theta) - b theta' + u."""
    th, om = x
    return np.array([om, (-M_REAL*G*L*np.sin(th) - B*om + u)
                     / (M_REAL*L**2)])


def energy(th, om, m=M_MODEL):
    """E = kinetic + potential, zero hanging at rest, 2 m g l at the top."""
    return 0.5*m*L**2*om**2 + m*G*L*(1 - np.cos(th))


E_TOP = 2*M_MODEL*G*L               # energy of the upright pendulum


def wrap(a):
    """Angle in [-pi, pi)."""
    return (a + np.pi) % (2*np.pi) - np.pi


# ---------------- the controllers ----------------
def pd_gravity(th, om, target):
    """[PDG] u = m g l sin(theta) - m l^2 (kp e + kd e'), e = theta - target."""
    e = wrap(th - target)
    return M_MODEL*G*L*np.sin(th) - M_MODEL*L**2*(KP*e + KD*om)


def controller(th, om, mem):
    if LAW == 'none':
        return 0.0
    if LAW == 'pdg':
        return float(np.clip(pd_gravity(th, om, TARGET), -3.0, 3.0))
    # [CATCH] after the switch (decided in simulate), balance at the top
    if mem['caught']:
        return float(np.clip(pd_gravity(th, om, np.pi), -U_MAX, U_MAX))
    # [SWING] pump energy: u = (b - k (E - E_top)) theta', then saturate
    e_tilde = energy(th, om) - E_TOP
    return float(np.clip((B - K_PUMP*e_tilde)*om, -U_MAX, U_MAX))


def lyapunov(th, om):
    """The function V(t) plotted for each law."""
    if LAW == 'pdg':                   # V = 1/2 m l^2 (e'^2 + kp e^2)
        e = wrap(th - TARGET)
        return 0.5*M_MODEL*L**2*(om**2 + KP*e**2)
    if LAW == 'swing':                 # V = 1/2 (E - E_top)^2
        return 0.5*(energy(th, om) - E_TOP)**2
    return energy(th, om, M_REAL)      # free pendulum: V = E


def simulate():
    x = np.array([THETA0, 0.0]); mem = {'caught': False, 't_catch': None}
    keep = round(1/(FPS*H)); out = []
    for k in range(int(T_END/H) + 1):
        t = k*H
        # [CATCH] switch once, between steps: near the top AND slow
        near = abs(wrap(x[0] - np.pi)) < 0.3 and abs(x[1]) < 3.0
        if LAW == 'swing' and CATCH and near and not mem['caught']:
            mem['caught'], mem['t_catch'] = True, t   # switching time resolved to within H
        u = controller(x[0], x[1], mem)
        if k % keep == 0:
            out.append([t, x[0], u, lyapunov(*x)])
        f = lambda z: plant(z, controller(z[0], z[1], mem))  # mode fixed
        k1 = f(x); k2 = f(x + H/2*k1); k3 = f(x + H/2*k2); k4 = f(x + H*k3)
        x = x + H/6*(k1 + 2*k2 + 2*k3 + k4)
    return mem['t_catch'], np.array(out).T


# ---------------- animation ----------------
def animate(t, th, u, V):
    fig = plt.figure(figsize=(10, 4.5))
    ax0 = fig.add_subplot(1, 2, 1)
    ax1 = fig.add_subplot(3, 2, 2); ax2 = fig.add_subplot(3, 2, 4)
    ax3 = fig.add_subplot(3, 2, 6)
    ax0.set_xlim(-0.4, 0.4); ax0.set_ylim(-0.4, 0.4)
    ax0.set_aspect('equal'); ax0.axis('off'); ax0.set_title(SCENARIO)
    ax0.plot(0, 0, 'ks', ms=6)
    rod, = ax0.plot([], [], lw=3, color='#1F6F8B')
    bob, = ax0.plot([], [], 'o', ms=14, color='#1F6F8B')
    label = ax0.text(-0.38, -0.36, '')
    vname = 'E [J]' if LAW == 'none' else 'V'
    curves = []
    for ax, y, lab, c in [(ax1, th, r'$\theta$ [rad]', '#1F6F8B'),
                          (ax2, V, vname, '#4C956C'),
                          (ax3, u, 'u [N.m]', '#C1666B')]:
        ax.set_xlim(0, T_END); pad = 0.1*(y.max() - y.min()) + 0.05
        ax.set_ylim(y.min() - pad, y.max() + pad); ax.set_ylabel(lab)
        curves.append(ax.plot([], [], color=c, lw=1.5)[0])
    ax3.set_xlabel('Time [s]'); fig.tight_layout()

    def update(k):
        x, y = L*np.sin(th[k]), -L*np.cos(th[k])
        rod.set_data([0, x], [0, y]); bob.set_data([x], [y])
        for c, y_ in zip(curves, (th, V, u)):
            c.set_data(t[:k+1], y_[:k+1])
        label.set_text(f't = {t[k]:.1f} s')
        return [rod, bob, label] + curves

    return FuncAnimation(fig, update, frames=len(t),
                         interval=1000/FPS, blit=True)


def save_summary(t, th, u, V):
    """Save angle, V and torque to <SCENARIO>.png, for the homework."""
    fig, axs = plt.subplots(3, 1, figsize=(6, 5), sharex=True)
    for ax, y, lab, c in [(axs[0], th, r'$\theta$ [rad]', '#1F6F8B'),
                          (axs[1], V, 'V', '#4C956C'),
                          (axs[2], u, 'u [N.m]', '#C1666B')]:
        ax.plot(t, y, color=c); ax.set_ylabel(lab); ax.grid(alpha=0.3)
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
    t_catch, (t, th, u, V) = simulate()
    dV = np.diff(V)
    print(f'{SCENARIO}: final angle {th[-1]:.3f} rad, final V {V[-1]:.2e},'
          f' largest increase of V in one frame {max(dV.max(), 0):.2e},'
          f' max |u| {np.abs(u).max():.2f} N.m')
    if LAW == 'swing':
        print('  caught at t =', 'never' if t_catch is None
              else f'{t_catch:.2f} s')
    save_summary(t, th, u, V)            # writes <SCENARIO>.png
    show(animate(t, th, u, V))
