import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt

# 1. Parameters
Nx = 40
Ny = 40

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


# 2. MAC Grid
# Pressure: cell centers
x_P = x_start + (np.arange(Nx) + 0.5) * dx
y_P = y_start + (np.arange(Ny) + 0.5) * dy

# U: vertical faces
# The periodic duplicate at x = 1 is not stored
x_U = x_start + np.arange(Nx) * dx
y_U = y_P.copy()

# V: horizontal faces
# Top and bottom boundary values are known
x_V = x_P.copy()
y_V = y_start + np.arange(1, Ny) * dy

# 3. Manufactured Analytical Solution
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


# 4. Forcing Functions
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


# 5. Indexing
#
# k = i + j*Nx
def idx_u(i, j):
    return (i % Nx) + j * Nx


def idx_v(i, j):
    return (i % Nx) + j * Nx


def idx_p(i, j):
    return (i % Nx) + j * Nx


# 6. U Laplacian
LU = sp.lil_matrix((nU, nU))

for j in range(Ny):
    for i in range(Nx):

        k = idx_u(i, j)

        # Periodic x-direction
        LU[k, idx_u(i - 1, j)] += 1.0 / dx**2
        LU[k, k] += -2.0 / dx**2
        LU[k, idx_u(i + 1, j)] += 1.0 / dx**2

        # y-direction
        # U = 0 on the top and bottom boundaries
        if j == 0:

            # Bottom boundary ghost-point treatment
            LU[k, k] += -3.0 / dy**2
            LU[k, idx_u(i, j + 1)] += 1.0 / dy**2

        elif j == Ny - 1:

            # Top boundary ghost-point treatment
            LU[k, idx_u(i, j - 1)] += 1.0 / dy**2
            LU[k, k] += -3.0 / dy**2

        else:

            LU[k, idx_u(i, j - 1)] += 1.0 / dy**2
            LU[k, k] += -2.0 / dy**2
            LU[k, idx_u(i, j + 1)] += 1.0 / dy**2


LU = LU.tocsr()


# 7. V Laplacian
LV = sp.lil_matrix((nV, nV))

# Boundary contributions to the RHS
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

        # South neighbor
        if j == 0:
            bc_V[k] -= V_bottom / dy**2
        else:
            LV[k, idx_v(i, j - 1)] += 1.0 / dy**2

        # North neighbor
        if j == Ny - 2:
            bc_V[k] -= V_top / dy**2
        else:
            LV[k, idx_v(i, j + 1)] += 1.0 / dy**2


LV = LV.tocsr()


# 8. Pressure Gradient Operators
Gx = sp.lil_matrix((nU, nP))
Gy = sp.lil_matrix((nV, nP))


# dP/dx at U locations
for j in range(Ny):
    for i in range(Nx):

        ku = idx_u(i, j)

        Gx[ku, idx_p(i, j)] += 1.0 / dx
        Gx[ku, idx_p(i - 1, j)] += -1.0 / dx


# dP/dy at V locations
for j in range(Ny - 1):
    for i in range(Nx):

        kv = idx_v(i, j)

        Gy[kv, idx_p(i, j + 1)] += 1.0 / dy
        Gy[kv, idx_p(i, j)] += -1.0 / dy


Gx = Gx.tocsr()
Gy = Gy.tocsr()


# 9. Divergence Operators
Dx = sp.lil_matrix((nP, nU))
Dy = sp.lil_matrix((nP, nV))

# Boundary contributions to the divergence equation
bc_div = np.zeros(nP)


for j in range(Ny):
    for i in range(Nx):

        kp = idx_p(i, j)

        # dU/dx = (U_east - U_west) / dx
        Dx[kp, idx_u(i + 1, j)] += 1.0 / dx
        Dx[kp, idx_u(i, j)] += -1.0 / dx

        # dV/dy = (V_north - V_south) / dy

        # North face
        if j == Ny - 1:
            bc_div[kp] -= V_top / dy
        else:
            Dy[kp, idx_v(i, j)] += 1.0 / dy

        # South face
        if j == 0:
            bc_div[kp] += V_bottom / dy
        else:
            Dy[kp, idx_v(i, j - 1)] += -1.0 / dy


Dx = Dx.tocsr()
Dy = Dy.tocsr()

# 10. Operator Checks
check_x = Gx + Dx.T
check_y = Gy + Dy.T

if check_x.nnz == 0:
    err_x = 0.0
else:
    err_x = np.max(np.abs(check_x.data))

if check_y.nnz == 0:
    err_y = 0.0
else:
    err_y = np.max(np.abs(check_y.data))


print("Operator checks")
print("----------------")
print("||Gx + Dx^T||_max =", err_x)
print("||Gy + Dy^T||_max =", err_y)

# 11. Right-Hand Side
XU, YU = np.meshgrid(x_U, y_U)
XV, YV = np.meshgrid(x_V, y_V)

f_rhs = f_force(XU, YU).ravel()
g_rhs = g_force(XV, YV).ravel()

# U boundary values are zero
f_bc = f_rhs.copy()

# Include known V boundary contributions
g_bc = g_rhs + mu * bc_V

# Divergence boundary contribution
h_bc = bc_div.copy()

b = np.concatenate(
    (
        f_bc,
        g_bc,
        h_bc
    )
)

# 12. Assemble the Stokes System
A = sp.bmat(
    [
        [mu * LU, None, -Gx],
        [None, mu * LV, -Gy],
        [Dx, Dy, None]
    ],
    format="lil"
)

# 13. Fix the Pressure Null Space
# Pressure is determined only up to an additive constant.
# Fix the first pressure unknown to zero.

pressure_index = nU + nV

A[pressure_index, :] = 0.0
A[pressure_index, pressure_index] = 1.0

b[pressure_index] = 0.0

A = A.tocsr()

# 14. Solve
solution = spla.spsolve(A, b)

U_sol = solution[:nU]

V_sol = solution[
    nU:nU + nV
]

P_sol = solution[
    nU + nV:
]

# 15. Reshape the Solution
U = U_sol.reshape(
    (Ny, Nx)
)

V_interior = V_sol.reshape(
    (Ny - 1, Nx)
)

P = P_sol.reshape(
    (Ny, Nx)
)


# Reconstruct the full V grid
V = np.empty(
    (Ny + 1, Nx)
)

V[0, :] = V_bottom

V[1:Ny, :] = V_interior

V[Ny, :] = V_top


# Pressure is only determined up to a constant
P = P - np.mean(P)

# 16. Interpolate Velocity to Cell Centers
# U is periodic in x
U_center = 0.5 * (
    U + np.roll(U, -1, axis=1)
)

# Average the north/south V faces
V_center = 0.5 * (
    V[:-1, :] + V[1:, :]
)

velocity_magnitude = np.sqrt(
    U_center**2 + V_center**2
)

# 17. Analytical Solution at Cell Centers
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

# 18. Error Checks
U_error = np.max(
    np.abs(U_center - U_an)
)

V_error = np.max(
    np.abs(V_center - V_an)
)

P_error = np.max(
    np.abs(P - P_an)
)


print()
print("Solution checks")
print("---------------")

print(
    f"U range: "
    f"{U.min():.6f} to {U.max():.6f}"
)

print(
    f"V range: "
    f"{V.min():.6f} to {V.max():.6f}"
)

print(
    f"|u| range: "
    f"{velocity_magnitude.min():.6f} "
    f"to {velocity_magnitude.max():.6f}"
)

print(
    f"Max U error: {U_error:.6e}"
)

print(
    f"Max V error: {V_error:.6e}"
)

print(
    f"Max P error: {P_error:.6e}"
)

# 19. Plot Velocity Magnitude and Streamlines
fig, ax = plt.subplots(
    figsize=(7, 7)
)

contour = ax.contourf(
    XP,
    YP,
    velocity_magnitude,
    levels=100,
    cmap="YlGnBu_r"
)

# Colorbar
cbar = fig.colorbar(
    contour,
    ax=ax
)

# Make colorbar tick labels larger
cbar.ax.tick_params(
    labelsize=13
)


# Streamlines
ax.streamplot(
    x_P,
    y_P,
    U_center,
    V_center,
    color="white",
    density=1.2,
    linewidth=1.2,
    arrowsize=1.2
)


# Axes
ax.set_xlim(
    x_start,
    x_end
)

ax.set_ylim(
    y_start,
    y_end
)

ax.set_xlabel(
    r"$x$",
    fontsize=18
)

ax.set_ylabel(
    r"$y$",
    fontsize=18
)

ax.tick_params(
    axis="both",
    labelsize=13
)


# Title
ax.set_title(
    r"$\|\vec{u}\|$",
    fontsize=24,
    pad=12
)


fig.tight_layout()


# Save figure for the LaTeX write-up
fig.savefig(
    "P1_velocity_magnitude.png",
    dpi=300,
    bbox_inches="tight"
)


plt.show()