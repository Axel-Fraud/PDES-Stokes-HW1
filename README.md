# MA 550 - Homework 1: Stokes Equations

This repository contains my solutions for MA 550 Homework 1, which focuses on solving the two-dimensional Stokes equations using a MAC staggered-grid finite-difference discretization.

## Problem 1: Manufactured Solution

`P1.py` solves the Stokes equations on a staggered grid using the prescribed manufactured solution. The numerical velocity and pressure solutions are compared with the analytical solutions to verify the implementation.

## Problem 2: Convergence Study

`P2.py` performs a grid-refinement study for increasing values of \(N_x=N_y\). Relative L2 errors are computed for U, V, and P, and the results demonstrate second-order convergence.

## Problem 3: Charlie's Rule

`P3.py` extends the Stokes solver to a lid-driven cavity problem. The top wall moves with unit horizontal velocity while the remaining walls are stationary. The resulting recirculating flow and vorticity field are visualized.

## Files

- `P1.py` - Manufactured-solution Stokes solver
- `P2.py` - Grid-refinement and convergence study
- `P3.py` - Lid-driven Stokes cavity
- `P1.png` - Problem 1 velocity visualization
- `P2.png` - Problem 2 convergence plot
- `P3_1.png` - Lid-driven cavity velocity and streamlines
- `P3_3.png` - Lid-driven cavity vorticity and streamlines
- `main.tex` - LaTeX source for the homework report
- `PDES_Stokes_HW1.pdf` - Final homework report

## Requirements

The Python scripts require:

- Python 3
- NumPy
- SciPy
- Matplotlib

## Running the Code

Each problem can be run independently from the repository directory:

```bash
python P1.py
python P2.py
python P3.py