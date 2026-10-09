import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt

# 1. Analytical Solutions and Forcing Functions
def U_exact(x, y):
    return (
        np.sin(2.0 * np.pi * y)
        * np.sin(2.0 * np.pi * x)
    )


def V_exact(x, y):
    return (
        -3.5
        + np.cos(2.0 * np.pi * x)
        * (np.cos(2.0 * np.pi * y) - 1.0)
    )


def P_exact(x, y):
    return (
        np.sin(2.0 * np.pi * y)
        * np.cos(2.0 * np.pi * x)
    )


def f_force(x, y):
    return (
        (2.0 * np.pi - 8.0 * np.pi**2)
        * np.sin(2.0 * np.pi * y)
        * np.sin(2.0 * np.pi * x)
    )


def g_force(x, y):
    return (
        -4.0 * np.pi**2
        * np.cos(2.0 * np.pi * x)
        * (2.0 * np.cos(2.0 * np.pi * y) - 1.0)
        - 2.0 * np.pi
        * np.cos(2.0 * np.pi * y)
        * np.cos(2.0 * np.pi * x)
    )

# 2. Stokes Solver
def solve_stokes(Nx, Ny):

    x_start = 0.0
    x_end = 1.0

    y_start = 0.0
    y_end = 1.0

    mu = 1.0

    dx = (x_end - x_start) / Nx
    dy = (y_end - y_start) / Ny

    V_bottom = -3.5
    V_top = -3.5

    nU = Nx * Ny
    nV = Nx * (Ny - 1)
    nP = Nx * Ny


    # MAC Grid
    x_P = x_start + (np.arange(Nx) + 0.5) * dx
    y_P = y_start + (np.arange(Ny) + 0.5) * dy

    x_U = x_start + np.arange(Nx) * dx
    y_U = y_P.copy()

    x_V = x_P.copy()
    y_V = y_start + np.arange(1, Ny) * dy

    # Indexing
    def idx_u(i, j):
        return (i % Nx) + j * Nx

    def idx_v(i, j):
        return (i % Nx) + j * Nx

    def idx_p(i, j):
        return (i % Nx) + j * Nx

    # U Laplacian
    LU = sp.lil_matrix((nU, nU))

    for j in range(Ny):
        for i in range(Nx):

            k = idx_u(i, j)

            # Periodic x-direction
            LU[k, idx_u(i - 1, j)] += 1.0 / dx**2
            LU[k, k] += -2.0 / dx**2
            LU[k, idx_u(i + 1, j)] += 1.0 / dx**2

            # y-direction
            if j == 0:

                LU[k, k] += -3.0 / dy**2
                LU[k, idx_u(i, j + 1)] += 1.0 / dy**2

            elif j == Ny - 1:

                LU[k, idx_u(i, j - 1)] += 1.0 / dy**2
                LU[k, k] += -3.0 / dy**2

            else:

                LU[k, idx_u(i, j - 1)] += 1.0 / dy**2
                LU[k, k] += -2.0 / dy**2
                LU[k, idx_u(i, j + 1)] += 1.0 / dy**2

    LU = LU.tocsr()

    # V Laplacian
    LV = sp.lil_matrix((nV, nV))

    bc_V = np.zeros(nV)

    for j in range(Ny - 1):
        for i in range(Nx):

            k = idx_v(i, j)

            # Periodic x-direction
            LV[k, idx_v(i - 1, j)] += 1.0 / dx**2
            LV[k, k] += -2.0 / dx**2
            LV[k, idx_v(i + 1, j)] += 1.0 / dx**2

            # y-direction
            LV[k, k] += -2.0 / dy**2

            if j == 0:
                bc_V[k] -= V_bottom / dy**2
            else:
                LV[k, idx_v(i, j - 1)] += 1.0 / dy**2

            if j == Ny - 2:
                bc_V[k] -= V_top / dy**2
            else:
                LV[k, idx_v(i, j + 1)] += 1.0 / dy**2

    LV = LV.tocsr()

    # Pressure Gradient Operators
    Gx = sp.lil_matrix((nU, nP))
    Gy = sp.lil_matrix((nV, nP))

    for j in range(Ny):
        for i in range(Nx):

            ku = idx_u(i, j)

            Gx[ku, idx_p(i, j)] += 1.0 / dx
            Gx[ku, idx_p(i - 1, j)] += -1.0 / dx


    for j in range(Ny - 1):
        for i in range(Nx):

            kv = idx_v(i, j)

            Gy[kv, idx_p(i, j + 1)] += 1.0 / dy
            Gy[kv, idx_p(i, j)] += -1.0 / dy

    Gx = Gx.tocsr()
    Gy = Gy.tocsr()

    # Divergence Operators
    Dx = sp.lil_matrix((nP, nU))
    Dy = sp.lil_matrix((nP, nV))

    bc_div = np.zeros(nP)

    for j in range(Ny):
        for i in range(Nx):

            kp = idx_p(i, j)

            # dU/dx
            Dx[kp, idx_u(i + 1, j)] += 1.0 / dx
            Dx[kp, idx_u(i, j)] += -1.0 / dx

            # dV/dy

            if j == Ny - 1:
                bc_div[kp] -= V_top / dy
            else:
                Dy[kp, idx_v(i, j)] += 1.0 / dy

            if j == 0:
                bc_div[kp] += V_bottom / dy
            else:
                Dy[kp, idx_v(i, j - 1)] += -1.0 / dy

    Dx = Dx.tocsr()
    Dy = Dy.tocsr()

    # Right-Hand Side
    XU, YU = np.meshgrid(x_U, y_U)
    XV, YV = np.meshgrid(x_V, y_V)

    f_rhs = f_force(XU, YU).ravel()
    g_rhs = g_force(XV, YV).ravel()

    f_bc = f_rhs.copy()
    g_bc = g_rhs + mu * bc_V
    h_bc = bc_div.copy()

    b = np.concatenate(
        (
            f_bc,
            g_bc,
            h_bc
        )
    )

    # Assemble Stokes System
    A = sp.bmat(
        [
            [mu * LU, None, -Gx],
            [None, mu * LV, -Gy],
            [Dx, Dy, None]
        ],
        format="lil"
    )

    # Fix Pressure Null Space
    pressure_index = nU + nV

    A[pressure_index, :] = 0.0
    A[pressure_index, pressure_index] = 1.0

    b[pressure_index] = 0.0

    A = A.tocsr()

    # Solve
    solution = spla.spsolve(A, b)

    U_sol = solution[:nU]

    V_sol = solution[
        nU:nU + nV
    ]

    P_sol = solution[
        nU + nV:
    ]

    # Reshape
    U = U_sol.reshape(
        (Ny, Nx)
    )

    V_interior = V_sol.reshape(
        (Ny - 1, Nx)
    )

    P = P_sol.reshape(
        (Ny, Nx)
    )


    # Full V grid including boundary values
    V = np.empty(
        (Ny + 1, Nx)
    )

    V[0, :] = V_bottom
    V[1:Ny, :] = V_interior
    V[Ny, :] = V_top


    # Remove pressure mean
    P = P - np.mean(P)

    # Interpolate U and V to Cell Centers
    U_center = 0.5 * (
        U + np.roll(U, -1, axis=1)
    )

    V_center = 0.5 * (
        V[:-1, :] + V[1:, :]
    )

    # Analytical Solution at Cell Centers
    XP, YP = np.meshgrid(
        x_P,
        y_P
    )

    U_an = U_exact(
        XP,
        YP
    )

    V_an = V_exact(
        XP,
        YP
    )

    P_an = P_exact(
        XP,
        YP
    )


    return (
        U_center,
        V_center,
        P,
        U_an,
        V_an,
        P_an
    )


# 3. Grid Refinement Study
N_values = np.array(
    [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
)

error_U = []
error_V = []
error_P = []


for N in N_values:

    print(f"Solving N = {N}...")

    (
        U_num,
        V_num,
        P_num,
        U_an,
        V_an,
        P_an
    ) = solve_stokes(N, N)


    # Relative L2 Errors
    err_U = (
        np.linalg.norm(U_num - U_an)
        / np.linalg.norm(U_an)
    )

    err_V = (
        np.linalg.norm(V_num - V_an)
        / np.linalg.norm(V_an)
    )

    err_P = (
        np.linalg.norm(P_num - P_an)
        / np.linalg.norm(P_an)
    )


    error_U.append(err_U)
    error_V.append(err_V)
    error_P.append(err_P)


error_U = np.array(error_U)
error_V = np.array(error_V)
error_P = np.array(error_P)


# 4. Observed Convergence Orders
h_values = 1.0 / N_values


def convergence_order(errors, h):
    return (
        np.log(errors[:-1] / errors[1:])
        / np.log(h[:-1] / h[1:])
    )


order_U = convergence_order(
    error_U,
    h_values
)

order_V = convergence_order(
    error_V,
    h_values
)

order_P = convergence_order(
    error_P,
    h_values
)


# 5. Print Error Table
print()
print("Relative L2 Errors")
print("------------------------------------------------------")
print(
    f"{'N':>5} "
    f"{'Error U':>14} "
    f"{'Error V':>14} "
    f"{'Error P':>14}"
)
print("------------------------------------------------------")


for i in range(len(N_values)):

    print(
        f"{N_values[i]:5d} "
        f"{error_U[i]:14.6e} "
        f"{error_V[i]:14.6e} "
        f"{error_P[i]:14.6e}"
    )


print()
print("Observed Convergence Orders")
print("-----------------------------------------------")
print(
    f"{'N1 -> N2':>12} "
    f"{'U':>10} "
    f"{'V':>10} "
    f"{'P':>10}"
)
print("-----------------------------------------------")


for i in range(len(order_U)):

    print(
        f"{N_values[i]:3d} -> {N_values[i + 1]:3d} "
        f"{order_U[i]:10.4f} "
        f"{order_V[i]:10.4f} "
        f"{order_P[i]:10.4f}"
    )

# 6. First- and Second-Order Reference Lines
# Scale the reference lines so they appear clearly on the
# same plot as the numerical errors.

first_order = (
    error_U[0]
    * (N_values[0] / N_values)
)

second_order = (
    error_U[0]
    * (N_values[0] / N_values)**2
)


# 7. Error Plot
fig, ax = plt.subplots(
    figsize=(8, 6)
)

# U error
ax.loglog(
    N_values,
    error_U,
    marker="o",
    linestyle="-",
    linewidth=2.0,
    markersize=11,
    color="teal",
    markerfacecolor="white",
    markeredgewidth=2.0,
    label=r"Error $U$"
)

# V error
ax.loglog(
    N_values,
    error_V,
    marker="^",
    linestyle="-",
    linewidth=2.0,
    markersize=7,
    color="red",
    markerfacecolor="white",
    markeredgewidth=2.0,
    label=r"Error $V$"
)

# P error
ax.loglog(
    N_values,
    error_P,
    marker="*",
    linestyle="--",
    linewidth=2.0,
    markersize=11,
    color="darkorange",
    label=r"Error $P$"
)

# First-order reference
ax.loglog(
    N_values,
    first_order,
    linestyle="--",
    linewidth=2.0,
    color="black",
    label="First Order Check"
)

# Second-order reference
ax.loglog(
    N_values,
    second_order,
    linestyle="-",
    linewidth=2.0,
    color="black",
    label="Second Order Check"
)

ax.set_xlabel(
    r"$N_x=N_y$",
    fontsize=15
)

ax.set_ylabel(
    "Relative Error",
    fontsize=15
)

ax.set_title(
    r"Error Plots for $U$, $V$, and $P$",
    fontsize=17
)

ax.grid(
    True,
    which="both",
    linestyle=":",
    alpha=0.4
)
ax.set_xticks([10, 20, 30, 40, 50, 60, 80, 100])
ax.set_xticklabels([10, 20, 30, 40, 50, 60, 80, 100])


ax.legend(
    fontsize=15,
    frameon=True
)

ax.tick_params(
    axis="both",
    which="major",
    labelsize=11
)

fig.tight_layout()

fig.savefig(
    "P2_error_convergence.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()