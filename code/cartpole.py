# cartpole.py: Session 4, the inverted pendulum on a cart
# Download: github.com/gfloresc/intelligent-control, folder code/
# Dr. Gerardo Flores, Texas A&M International University
#
# Same pendulum as pendulum_inverted.py (m = 0.5 kg, l = 0.3 m,
# b = 0.05), now mounted on a cart of mass M = 1 kg. The motor
# pushes the CART with a force F; nothing acts on the pendulum.
# Pick a class example below, run, and watch the result.
#
# MARKS. The cart-pole slides have grey boxes with a mark.
# Search this file for it:
#   [CP-MODEL]  the nonlinear model
#   [CP-LIN]    the linearization A, B
#   [CP-DESIGN] the gain K from the chosen poles
#   [CP-NOISE]  noisy encoders and the filter

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.signal import place_poles

# ===================== CLASS EXAMPLES =====================
SCENARIOS = {
    'design':       dict(),                  # back to the top in ~3 s
    'angle_only':   dict(angle_only=True),   # pole up, cart drifts
    'heavy_pole':   dict(m=0.75),            # 50% heavier: still works
    'far_start':    dict(tilt=0.6),          # too far for a 10 N motor:
                                             # falls (tilt=0.5 still works)
    'push_cart':    dict(d=0.5),             # cart settles off center
    'push_integral': dict(d=0.5, integral=True),  # back to x = 0
    'noise_raw':    dict(noise=True),        # force jitters a lot
    'noise_filtered': dict(noise=True, tau=0.05),  # smooth again

    # --- add your own scenarios below ---
    'my_test':      dict(m=0.6, tilt=0.3),
}

SCENARIO = 'design'     # <- change this name

_s = SCENARIOS[SCENARIO]
M_REAL = _s.get('m', 0.5)           # real pendulum mass [kg]
TILT = _s.get('tilt', 0.2)          # initial tilt from upright [rad]
D_PUSH = _s.get('d', 0.0)           # constant force on cart [N], t >= 1 s
ANGLE_ONLY = _s.get('angle_only', False)  # ignore the cart states
INTEGRAL = _s.get('integral', False)      # integral action on x
NOISE = _s.get('noise', False)      # noisy encoders
TAU = _s.get('tau', None)           # velocity filter [s] (None = off)

# ---------------- fixed data ----------------
M, M_DESIGN, L, G, B = 1.0, 0.5, 0.3, 9.81, 0.05
POLES = [-2, -3, -4, -5]            # chosen closed-loop poles
POLE_Z = -1.5                       # extra pole for the integral
SIG_X, SIG_TH = 0.001, 0.01         # encoder noise [m], [rad]
F_MAX = 10.0                        # motor limit [N] (saturation, Session 3)
TS, H = 0.01, 0.001                 # sample time, plant step [s]
T_END, FPS = 8.0, 30


# ---------------- [CP-MODEL] nonlinear model ----------------
def plant(s, F, t, m=None):
    """s = [x, xdot, theta, thetadot], theta from straight down."""
    m = M_REAL if m is None else m
    x, xd, th, thd = s
    F = F + (D_PUSH if t >= 1.0 else 0.0)    # push on the cart
    sn, cs = np.sin(th), np.cos(th)
    # (M+m) x'' + m l cos(th) th'' - m l sin(th) th'^2 = F
    # m l^2 th'' + m l cos(th) x'' + m g l sin(th) + b th' = 0
    # solved for x'' and th'' (2 equations, 2 unknowns):
    den = M + m*sn**2
    xdd = (F + m*L*sn*thd**2 + m*G*sn*cs + B*cs*thd/L) / den
    thdd = -(cs*xdd + G*sn + B*thd/(m*L)) / L
    return np.array([xd, xdd, thd, thdd])


# ---------------- [CP-LIN] linearization at the top ----------------
def linear_model(m=M_DESIGN):
    """A, B at x = 0, theta = pi, at rest; d_theta = theta - pi."""
    A = np.array([[0, 1, 0, 0],
                  [0, 0, m*G/M, -B/(M*L)],
                  [0, 0, 0, 1],
                  [0, 0, (M + m)*G/(M*L), -(M + m)*B/(M*m*L**2)]])
    Bv = np.array([[0], [1/M], [0], [1/(M*L)]])
    return A, Bv


# ---------------- [CP-DESIGN] the gain K ----------------
def design():
    """Place the poles; with INTEGRAL, add z = integral of x."""
    A, Bv = linear_model()
    if INTEGRAL:                       # 5th state: z' = x
        A = np.block([[A, np.zeros((4, 1))],
                      [np.array([[1, 0, 0, 0, 0]])]])
        Bv = np.vstack([Bv, [[0]]])
        return place_poles(A, Bv, POLES + [POLE_Z]).gain_matrix[0]
    K = place_poles(A, Bv, POLES).gain_matrix[0]
    if ANGLE_ONLY:
        K[0] = K[1] = 0.0              # forget the cart states
    return K


# ---------------- [CP-NOISE] sensors and the controller ----------------
def controller(meas, mem, K):
    """meas = [x, theta] from the encoders; velocities estimated."""
    e = np.array([meas[0], meas[1] - np.pi])        # x, d_theta
    w = (e - mem['old']) / TS                       # raw slopes
    mem['old'] = e
    if TAU is not None:                             # low-pass filter
        mem['wf'] += (TS/TAU) * (w - mem['wf'])
        w = mem['wf']
    state = np.array([e[0], w[0], e[1], w[1]])
    if INTEGRAL:
        mem['z'] += e[0]*TS                         # z_k = z_{k-1} + Ts x_k
        state = np.append(state, mem['z'])
    return -float(K @ state)


def simulate():
    rng = np.random.default_rng(1)
    K = design()
    s = np.array([0.0, 0.0, np.pi - TILT, 0.0])
    mem = {'old': np.array([0.0, -TILT]), 'wf': np.zeros(2), 'z': 0.0}
    every, keep = round(TS/H), round(1/(FPS*H))
    out, F = [], 0.0
    for k in range(int(T_END/H) + 1):
        t = k*H
        if k % every == 0:
            meas = s[[0, 2]].copy()
            if NOISE:
                meas += [SIG_X, SIG_TH] * rng.standard_normal(2)
            F = controller(meas, mem, K)
            F = float(np.clip(F, -F_MAX, F_MAX))   # motor limit
        if k % keep == 0:
            out.append([t, s[0], s[2], F])
        k1 = plant(s, F, t); k2 = plant(s + H/2*k1, F, t + H/2)
        k3 = plant(s + H/2*k2, F, t + H/2); k4 = plant(s + H*k3, F, t + H)
        s = s + H/6*(k1 + 2*k2 + 2*k3 + k4)
    return K, np.array(out).T


# ---------------- animation ----------------
def animate(t, x, th, F):
    fig = plt.figure(figsize=(10, 4.5))
    ax0 = fig.add_subplot(1, 2, 1)
    ax1 = fig.add_subplot(3, 2, 2); ax2 = fig.add_subplot(3, 2, 4)
    ax3 = fig.add_subplot(3, 2, 6)
    span = min(max(1.0, np.abs(x).max() + 0.5), 3.0)
    ax0.set_xlim(-span, span); ax0.set_ylim(-0.3, 0.6); ax0.set_yticks([])
    ax0.set_aspect('equal'); ax0.axhline(-0.05, color='k', lw=1)
    ax0.set_title(SCENARIO)
    cart = plt.Rectangle((0, 0), 0.24, 0.1, color='#55606E')
    ax0.add_patch(cart)
    rod, = ax0.plot([], [], lw=3, color='#1F6F8B')
    bob, = ax0.plot([], [], 'o', ms=12, color='#1F6F8B')
    label = ax0.text(-span + 0.05, 0.5, '')
    ax0.set_xlabel('x [m]')
    curves = []
    for ax, y, lab, c in [(ax1, x, 'x [m]', '#55606E'),
                          (ax2, th - np.pi, r'$\theta-\pi$', '#1F6F8B'),
                          (ax3, F, 'F [N]', '#C1666B')]:
        ax.set_xlim(0, T_END); ax.axhline(0, color='k', ls=':', lw=1)
        pad = 0.1*(y.max() - y.min()) + 0.05
        ax.set_ylim(y.min() - pad, y.max() + pad); ax.set_ylabel(lab)
        curves.append(ax.plot([], [], color=c, lw=1.5)[0])
    ax3.set_xlabel('t [s]'); fig.tight_layout()

    def update(k):
        cart.set_xy((x[k] - 0.12, 0.0))
        px, py = x[k] + L*np.sin(th[k]), 0.1 - L*np.cos(th[k])
        rod.set_data([x[k], px], [0.1, py]); bob.set_data([px], [py])
        for c, y in zip(curves, (x, th - np.pi, F)):
            c.set_data(t[:k+1], y[:k+1])
        label.set_text(f't = {t[k]:.1f} s')
        return [cart, rod, bob, label] + curves

    return FuncAnimation(fig, update, frames=len(t),
                         interval=1000/FPS, blit=True)


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
    K, (t, x, th, F) = simulate()
    print(f'{SCENARIO}: K = {np.round(K, 2)}')
    print(f'  final x {x[-1]:+.3f} m, final tilt {th[-1] - np.pi:+.3f} rad,'
          f' force jitter after 4 s {F[t > 4].std():.2f} N')
    show(animate(t, x, th, F))
