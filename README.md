# SENG 5360 — Intelligent Control

Lecture slides, and code for **SENG 5360, Intelligent Control**, a graduate course in the
M.S. in Systems Engineering at **Texas A&M International University**.

Taught by **Dr. Gerardo Flores** · [RAPTOR Lab]

---

## About the course

SENG 5360 provides a foundation in the modeling and control of intelligent engineering systems, with
emphasis on model-based control strategies for nonlinear and uncertain dynamical systems of the kind
found in automation, robotics and cyber-physical systems. The course balances theoretical
development with engineering applications.

No prior course in control systems is assumed. Calculus, ordinary differential equations and linear
algebra are.

A single running example, the **actuated pendulum**, is carried through every session: it is
nonlinear, it has two equilibria with opposite behaviour, it can be linearized and seen to fail, its
parameters are uncertain, and part of its state is typically unmeasured.

---

## Modules

**1. Foundations, modeling and analysis of nonlinear engineering systems**
- review of essential linear systems concepts: state-space representations, eigenvalues, stability
- nonlinear state-space representations
- equilibrium points and local linearization
- parametric uncertainty and external disturbances
- measurement noise

**2. Nonlinear control and stability**
- regulation and tracking control problems
- basic nonlinear feedback design
- introductory Lyapunov stability analysis
- Lyapunov-based controller design
- practical robustness under uncertainty and external disturbances

**3. Estimation and adaptation in intelligent control**
- state estimation and disturbance estimation concepts
- observer-based control architectures
- unknown and changing system parameters
- introductory online parameter adaptation and simple adaptive-control structures
- brief overview of data-driven and learning-based methods for estimation and control

Applications from robotics, UAVs, automation, mechatronics and cyber-physical systems are integrated
throughout.

---

## Sessions

| # | Session | Module | Slides | Status |
|---|---|---|---|---|
| 1 | Course overview and state-space representations | 1 | `seng5360_s01.pdf` | available |
| 2 | Linear systems review | 1 | `seng5360_s02.pdf` | available |
| 3 | Equilibrium points and local linearization | 1 | `seng5360_s03.pdf` | available |
| 4 | Uncertainty, disturbances and measurement noise | 1 | | in preparation |
| 5 | Regulation and tracking; the error dynamics | 2 | | in preparation |
| 6 | Lyapunov stability: the direct method | 2 | | in preparation |
| 7 | LaSalle and asymptotic stability | 2 | | in preparation |
| 8 | Lyapunov-based controller design | 2 | | in preparation |
| 9 | Practical robustness | 2 | | in preparation |
| 10 | Observers and state estimation | 3 | | in preparation |
| 11 | Disturbance estimation | 3 | | in preparation |
| 12 | Observer-based control architectures | 3 | | in preparation |
| 13 | Unknown parameters and online adaptation | 3 | | in preparation |
| 14 | Adaptive structures; data-driven methods | 3 | | in preparation |

---

## Repository layout

```
.
├── slides/     compiled lecture slides, one PDF per session
└── code/       one Colab notebook per session, with every snippet
                shown on the slides of that session
```

---

## Notebooks

Each session has one Colab notebook in `code/`, containing every code snippet shown on the slides of
that session, in the order they appear. The listings on a slide carry a code icon and a number, for
example **Code 2.3**; the notebook uses the same numbering, so you can go straight to the cell you
need.

| Session | Notebook |
|---|---|
| 1 | [`session1_lab1.ipynb`](code/SENG5360_Session1_Lab1_Colab.ipynb) |
| 2 | [`session2_linear_toolbox.ipynb`](code/SENG5360_Session2_Linear_Toolbox_Colab.ipynb) |

The notebooks run in Google Colab with no setup. To run them locally you need `numpy`, `scipy` and
`matplotlib`.

Some cells are deliberately incomplete: they are the starting point of a homework item, not a
finished solution.

### Running the code

```bash
pip install numpy scipy matplotlib
python code/s3_01_lab_3_linearize_design_verify.py
```


## References

- **Course text:** J.-J. E. Slotine and W. Li, *Applied Nonlinear Control*, Prentice Hall, 1991.
- **Reference:** H. K. Khalil, *Nonlinear Systems*, 3rd ed., Prentice Hall, 2002.
- **Linear background:** K. J. Åström and R. M. Murray, *Feedback Systems*, Princeton University
  Press, 2008. Freely available online.
- **Linear systems, in depth:** C.-T. Chen, *Linear System Theory and Design*, 4th ed.,
  Oxford University Press, 2013.

---

## Using this material

These materials are shared for students of the course and for anyone teaching or learning nonlinear
control. If you use them, an acknowledgement is appreciated.

Corrections and issue reports are welcome. Several numbers on the slides come from simulations
included in this repository, so any discrepancy is reproducible and worth reporting.


---

## Contact

Dr. Gerardo Flores · Associate Professor, Texas A&M International University
Email: gerardo.flores@tamiu.edu · Phone: (956) 326-3297
