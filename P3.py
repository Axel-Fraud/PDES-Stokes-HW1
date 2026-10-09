import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt


# Lid-Driven Stokes Cavity
# 1. Parameters
Nx = 50
Ny = 50

x_start = 0.0
x_end = 1.0

y_start = 0.0
y_end = 1.0

mu = 1.0

dx = (x_end - x_start) / Nx
dy = (y_end - y_start) / Ny


# 2. Boundary Conditions
# Horizontal velocity U
U_left = 0.0
U_right = 0.0
U_bottom = 0.0
U_top = 1.0

# Vertical velocity V
V_left = 0.0
V_right = 0.0
V_bottom = 0.0
V_top = 0.0

# 3. Number of Unknowns
# U is stored on vertical faces.
# Left and right boundary values are known.
nU = (Nx - 1) * Ny

# V is stored on horizontal faces.
# Bottom and top boundary values are known.
nV = Nx * (Ny - 1)

# Pressure is stored at cell centers.
nP = Nx * Ny

n_total = nU + nV + nP

print("Grid information")
print("----------------")
print(f"Nx = {Nx}")
print(f"Ny = {Ny}")
print(f"U unknowns = {nU}")
print(f"V unknowns = {nV}")
print(f"P unknowns = {nP}")
print(f"Total unknowns = {n_total}")


# 4. Indexing Functions
def idx_u(i, j):
    """
    U unknown:
        i = 0,...,Nx-2
        j = 0,...,Ny-1

    Physical U face is i+1.
    """
    return i + j * (Nx - 1)


def idx_v(i, j):
    """
    V unknown:
        i = 0,...,Nx-1
        j = 0,...,Ny-2

    Physical V face is j+1.
    """
    return i + j * Nx


def idx_p(i, j):
    """
    Pressure unknown:
        i = 0,...,Nx-1
        j = 0,...,Ny-1
    """
    return i + j * Nx


# 5. U Laplacian
LU = sp.lil_matrix((nU, nU))

bc_U = np.zeros(nU)


for j in range(Ny):

    for i in range(Nx - 1):

        k = idx_u(i, j)

        # x-direction
        LU[k, k] += -2.0 / dx**2

        # West neighbor
        if i == 0:

            # Known U value on left wall
            bc_U[k] -= U_left / dx**2

        else:

            LU[k, idx_u(i - 1, j)] += 1.0 / dx**2


        # East neighbor
        if i == Nx - 2:

            # Known U value on right wall
            bc_U[k] -= U_right / dx**2

        else:

            LU[k, idx_u(i + 1, j)] += 1.0 / dx**2


        # y-direction
        # U points are half a grid spacing from the top and
        # bottom walls, so ghost points are used.

        if j == 0:

            # Bottom:
            #
            # U_ghost = 2*U_bottom - U_P
            #
            # This changes the diagonal coefficient from
            # -2 to -3.

            LU[k, k] += -3.0 / dy**2
            LU[k, idx_u(i, j + 1)] += 1.0 / dy**2

            bc_U[k] -= 2.0 * U_bottom / dy**2


        elif j == Ny - 1:

            # Top moving lid

            LU[k, idx_u(i, j - 1)] += 1.0 / dy**2
            LU[k, k] += -3.0 / dy**2

            bc_U[k] -= 2.0 * U_top / dy**2


        else:

            # Interior

            LU[k, idx_u(i, j - 1)] += 1.0 / dy**2
            LU[k, k] += -2.0 / dy**2
            LU[k, idx_u(i, j + 1)] += 1.0 / dy**2


LU = LU.tocsr()

# 6. V Laplacian
LV = sp.lil_matrix((nV, nV))

bc_V = np.zeros(nV)


for j in range(Ny - 1):

    for i in range(Nx):

        k = idx_v(i, j)

        # x-direction
        # V points are half a grid spacing from the left and
        # right walls, so ghost points are used.
        if i == 0:

            # Left wall

            LV[k, k] += -3.0 / dx**2
            LV[k, idx_v(i + 1, j)] += 1.0 / dx**2

            bc_V[k] -= 2.0 * V_left / dx**2


        elif i == Nx - 1:

            # Right wall

            LV[k, idx_v(i - 1, j)] += 1.0 / dx**2
            LV[k, k] += -3.0 / dx**2

            bc_V[k] -= 2.0 * V_right / dx**2


        else:

            # Interior

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

# 7. Pressure Gradient Operators
Gx = sp.lil_matrix((nU, nP))
Gy = sp.lil_matrix((nV, nP))


# dP/dx at U locations
for j in range(Ny):

    for i in range(Nx - 1):

        ku = idx_u(i, j)

        # Physical U face lies between pressure cells i and i+1

        Gx[ku, idx_p(i + 1, j)] += 1.0 / dx
        Gx[ku, idx_p(i, j)] += -1.0 / dx


# dP/dy at V locations
for j in range(Ny - 1):

    for i in range(Nx):

        kv = idx_v(i, j)

        # Physical V face lies between pressure rows j and j+1

        Gy[kv, idx_p(i, j + 1)] += 1.0 / dy
        Gy[kv, idx_p(i, j)] += -1.0 / dy


Gx = Gx.tocsr()
Gy = Gy.tocsr()

# 8. Divergence Operators
Dx = sp.lil_matrix((nP, nU))
Dy = sp.lil_matrix((nP, nV))

bc_div = np.zeros(nP)


for j in range(Ny):

    for i in range(Nx):

        kp = idx_p(i, j)

        # dU/dx
        # East U face
        if i == Nx - 1:

            bc_div[kp] -= U_right / dx

        else:

            Dx[kp, idx_u(i, j)] += 1.0 / dx


        # West U face
        if i == 0:

            bc_div[kp] += U_left / dx

        else:

            Dx[kp, idx_u(i - 1, j)] += -1.0 / dx

        # dV/dy
        # North V face
        if j == Ny - 1:

            bc_div[kp] -= V_top / dy

        else:

            Dy[kp, idx_v(i, j)] += 1.0 / dy


        # South V face
        if j == 0:

            bc_div[kp] += V_bottom / dy

        else:

            Dy[kp, idx_v(i, j - 1)] += -1.0 / dy


Dx = Dx.tocsr()
Dy = Dy.tocsr()

# 9. Operator Checks
check_x = Gx + Dx.T
check_y = Gy + Dy.T


if check_x.nnz == 0:

    err_x = 0.0

else:

    err_x = np.max(
        np.abs(check_x.data)
    )


if check_y.nnz == 0:

    err_y = 0.0

else:

    err_y = np.max(
        np.abs(check_y.data)
    )


print()
print("Operator checks")
print("----------------")

print(
    "||Gx + Dx^T||_max =",
    err_x
)

print(
    "||Gy + Dy^T||_max =",
    err_y
)

# 10. Right-Hand Side
# No body force for the lid-driven cavity.
# The moving wall is what drives the fluid.

f_rhs = np.zeros(nU)
g_rhs = np.zeros(nV)

f_bc = f_rhs + mu * bc_U
g_bc = g_rhs + mu * bc_V

h_bc = bc_div.copy()


b = np.concatenate(
    (
        f_bc,
        g_bc,
        h_bc
    )
)

# 11. Assemble Stokes System
A = sp.bmat(
    [
        [mu * LU, None, -Gx],
        [None, mu * LV, -Gy],
        [Dx, Dy, None]
    ],
    format="lil"
)

# 12. Pressure Gauge
# Pressure is determined only up to an additive constant.
# Fix one pressure value to zero.

pressure_index = nU + nV

A[pressure_index, :] = 0.0
A[pressure_index, pressure_index] = 1.0

b[pressure_index] = 0.0


A = A.tocsr()

# 13. Solve
print()
print("Solving lid-driven cavity...")


solution = spla.spsolve(
    A,
    b
)


U_sol = solution[
    :nU
]

V_sol = solution[
    nU:nU + nV
]

P_sol = solution[
    nU + nV:
]


print("Solve complete.")

# 14. Reconstruct Full Staggered Velocity Grids
# Full U grid
# Ny rows by Nx+1 vertical faces

U = np.zeros(
    (Ny, Nx + 1)
)

U[:, 0] = U_left
U[:, Nx] = U_right

U[:, 1:Nx] = U_sol.reshape(
    (Ny, Nx - 1)
)


# Full V grid
# Ny+1 horizontal faces by Nx columns
V = np.zeros(
    (Ny + 1, Nx)
)

V[0, :] = V_bottom
V[Ny, :] = V_top

V[1:Ny, :] = V_sol.reshape(
    (Ny - 1, Nx)
)


# Pressure
P = P_sol.reshape(
    (Ny, Nx)
)

P = P - np.mean(P)

# 15. Interpolate Velocity to Cell Centers
U_center = 0.5 * (
    U[:, :-1]
    + U[:, 1:]
)

V_center = 0.5 * (
    V[:-1, :]
    + V[1:, :]
)


velocity_magnitude = np.sqrt(
    U_center**2
    + V_center**2
)

# 16. Cell-Center Grid
x_P = (
    x_start
    + (np.arange(Nx) + 0.5) * dx
)

y_P = (
    y_start
    + (np.arange(Ny) + 0.5) * dy
)


XP, YP = np.meshgrid(
    x_P,
    y_P
)

# 17. Solution Checks
divergence = (
    Dx @ U_sol
    + Dy @ V_sol
    - h_bc
)


print()
print("Solution checks")
print("----------------")

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
    "Maximum divergence:",
    np.max(np.abs(divergence))
)

# 18. Velocity Magnitude and Streamlines
fig, ax = plt.subplots(
    figsize=(8, 7)
)


contour = ax.contourf(
    XP,
    YP,
    velocity_magnitude,
    levels=100,
    cmap="viridis"
)


cbar = fig.colorbar(
    contour,
    ax=ax
)

cbar.ax.tick_params(
    labelsize=14
)


ax.streamplot(
    x_P,
    y_P,
    U_center,
    V_center,
    color="white",
    density=1.7,
    linewidth=1.0,
    arrowsize=1.2
)


ax.set_xlim(
    x_start,
    x_end
)

ax.set_ylim(
    y_start,
    y_end
)

ax.set_aspect(
    "equal"
)


ax.set_xlabel(
    "x",
    fontsize=17
)

ax.set_ylabel(
    "y",
    fontsize=17
)

ax.set_title(
    "Lid-Driven Stokes Cavity",
    fontsize=20,
    pad=10
)

ax.tick_params(
    axis="both",
    which="major",
    labelsize=14
)


fig.tight_layout()


fig.savefig(
    "P3_lid_driven_cavity.png",
    dpi=300,
    bbox_inches="tight"
)


plt.show()

# 19. Vorticity Field
# For 2D flow:
# omega = dV/dx - dU/dy
# U_center and V_center are both defined at pressure-cell centers.

dV_dx = np.gradient(
    V_center,
    dx,
    axis=1
)

dU_dy = np.gradient(
    U_center,
    dy,
    axis=0
)

vorticity = dV_dx - dU_dy


fig, ax = plt.subplots(
    figsize=(8, 7)
)

vort_limit = np.percentile(
    np.abs(vorticity),
    98
)

levels = np.linspace(
    -vort_limit,
    vort_limit,
    100
)

vort_plot = ax.contourf(
    XP,
    YP,
    vorticity,
    levels=levels,
    cmap="RdBu_r",
    extend="both"
)

cbar = fig.colorbar(
    vort_plot,
    ax=ax
)

# Make colorbar tick numbers larger
cbar.ax.tick_params(
    labelsize=14
)


# Add streamlines so rotation can be seen together
# with the velocity field.

ax.streamplot(
    x_P,
    y_P,
    U_center,
    V_center,
    color="black",
    density=1.2,
    linewidth=0.7,
    arrowsize=0.9
)

ax.set_xlim(
    x_start,
    x_end
)

ax.set_ylim(
    y_start,
    y_end
)

ax.set_aspect(
    "equal"
)

ax.set_xlabel(
    "x",
    fontsize=18
)

ax.set_ylabel(
    "y",
    fontsize=18
)

ax.set_title(
    "Vorticity Field for Lid-Driven Stokes Cavity",
    fontsize=20,
    pad=12
)

# Make x- and y-axis tick numbers larger
ax.tick_params(
    axis="both",
    which="major",
    labelsize=14
)

fig.tight_layout()

fig.savefig(
    "P3_vorticity.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# 20. Pressure Field
# The discontinuous velocity condition at the upper corners
# produces very large pressure gradients there.
# Use a percentile-based plotting range so the interior
# pressure structure remains visible. This ONLY changes the
# visualization; it does not modify the numerical solution.

pressure_limit = np.percentile(
    np.abs(P),
    97
)

pressure_levels = np.linspace(
    -pressure_limit,
    pressure_limit,
    100
)


fig, ax = plt.subplots(
    figsize=(8, 7)
)

pressure_plot = ax.contourf(
    XP,
    YP,
    P,
    levels=pressure_levels,
    cmap="RdBu_r",
    extend="both"
)

cbar = fig.colorbar(
    pressure_plot,
    ax=ax
)

# Make colorbar tick numbers larger
cbar.ax.tick_params(
    labelsize=14
)

ax.set_xlim(
    x_start,
    x_end
)

ax.set_ylim(
    y_start,
    y_end
)

ax.set_aspect(
    "equal"
)

ax.set_xlabel(
    "x",
    fontsize=18
)

ax.set_ylabel(
    "y",
    fontsize=18
)

ax.set_title(
    "Pressure Field for Lid-Driven Stokes Cavity",
    fontsize=20,
    pad=12
)

# Make x- and y-axis tick numbers larger
ax.tick_params(
    axis="both",
    which="major",
    labelsize=14
)

fig.tight_layout()

fig.savefig(
    "P3_pressure.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()