import numpy as np
from scipy.integrate import solve_ivp
from SCBE import SCBE_diff_eq
from energyband import energyband
from A_fun import A_fun
from F_fun import F_fun
from v_band import v_band
import matplotlib.pyplot as plt

# Setup parameters (Copied from Mads)
F_0 = 0.0040
omega = 0.014
T_0 = 2 * np.pi / omega
n_c = 10
T_c = T_0 * n_c / np.pi
b_m = 0  # Band model

a = 2.46 / 0.5291  # Where do these come from?
ax = np.sqrt(3) * a / 2
ay = 3 / 4 * a

dx = lambda k: np.sqrt(0.302 / np.abs(energyband(k, b_m, 1) - energyband(k, b_m, 0))) * 3
dy = lambda k: np.sqrt(0.302 / np.abs(energyband(k, b_m, 1) - energyband(k, b_m, 0))) * 3
dz = lambda k: np.sqrt(0.375 / (energyband(k, b_m, 1) - energyband(k, b_m, 0)))  # is assumed to be k-independent

# Creating k-grid
n    = 3
kx_n = n  # Square grid for simplicity
ky_n = n
scale = 1
kx = np.linspace(-scale * np.pi, scale * np.pi, kx_n) / ax
ky = np.linspace(-scale * np.pi, scale * np.pi, ky_n) / ay
d2k = 2 * np.pi / (a * kx_n) * 2 * np.pi / (a * ky_n)  # Infiniticimal area element in k-space

# Starting conditions and time
tmax = n_c * T_0
tspan = np.linspace(0, tmax, 20000)
y0 = np.array([0, 1, 0, 0])  # Only the valence bond is populated

# Setting up currents
jx_ra = np.zeros(tspan.shape[0])
jy_ra = np.zeros(tspan.shape[0])
jx_er = np.zeros(tspan.shape[0])
jy_er = np.zeros(tspan.shape[0])

# Parameters for ODE
F_m = 1  # Field model (1 = lin. pol. in x)

A = lambda t: A_fun(t, F_m, F_0)
F = lambda t: F_fun(t, F_m, F_0)
reltol = 1e-8
abstol = 1e-10

# Pugging into ODE
for j in range(ky_n):
    ky_j = ky[j]
    for n in range(kx_n):
        print(f"j,n = {j},{n}")
        k = np.array([kx[n], ky_j, 0])
        dxy = np.array([dx(k), dy(k)])
        sol = solve_ivp(SCBE_diff_eq, [0, tmax], y0, t_eval=tspan, args=(0, T_0 / 4, k, F, A, b_m), rtol=reltol,
                        atol=abstol)
        t = sol.t  # Unpacking t's
        y = sol.y  # Unpacking y's
        vv = np.real(v_band(k + A(t), b_m, 0))  # v in valence band
        vc = np.real(v_band(k + A(t), b_m, 1))  # v in conduction band
        # print(f" y[0].shape = {y[0].shape}")
        # print(f" np.exp(1j*y[3]).shape = {np.exp(1j*y[3]).shape}")
        # print(f" vv[:,0:2].shape = {vv[:,0].shape}")
        # print(np.multiply(y[0],vv[:,0]).shape)
        jx_ra = jx_ra + (y[1] * vv[:, 0] + y[2] * vc[:, 0]) * d2k
        jy_ra = jy_ra + (y[1] * vv[:, 1] + y[2] * vc[:, 1]) * d2k
        jx_er = jx_er + (dxy[0] * y[0] * np.exp(1j * y[3]) + dxy[0] * np.conj(y[0]) * np.exp(-1j * y[3])) * d2k
        jy_er = jy_er + (dxy[1] * y[0] * np.exp(1j * y[3]) + dxy[1] * np.conj(y[0]) * np.exp(-1j * y[3])) * d2k
        # j_er = j_er + (dxy * y[0].dot(np.exp(1j*y[3]))  + dxy * np.conj(y[0]).dot(np.exp(-1j * y[3])))*d2k

# Plotting currents
j_ra_tot = jx_ra + jy_ra
j_er_tot = jx_er + jy_er


plt.subplot(2, 1, 1)
plt.plot(tspan, np.real(j_er_tot))
plt.xlabel("t")
plt.title("Real(J_er)")

plt.subplot(2, 1, 2)
plt.plot(tspan, np.real(j_ra_tot))
plt.title("Real(J_ra)")
plt.xlabel("t")


plt.tight_layout()
plt.show()
