import numpy as np
from energyband import energyband


def SCBE_diff_eq(t, y, dephase, T2, k, F_fun, A_fun, d, i_m):
    # y should be a vector with: [pi, val, cond, diff_for_exponent]
    Omega = np.dot(F_fun(t), d)
    # print(f"Omega = {Omega}")
    E_gab = energyband(k + A_fun(t), i_m, 1) - energyband(k + A_fun(t), i_m, 0)  # i_m referring to the specific band structure for a given material.
    if dephase == 0:
        ## Don't know what to do with internal xi - (xi_c-xi_v)
        dy1dt = -1j * Omega * (y[1] - y[2]) * np.exp(-1j * y[3])
        dy2dt = -1j * y[0] * np.conj(Omega) * np.exp(1j * y[3]) + 1j * np.conj(y[0]) *Omega * np.exp(-1j * y[3])
        dy3dt = -dy2dt
        dy4dt = E_gab

    elif dephase == 1:
        dy1dt = -y[0] / T2 - 1j * Omega * (y[1] - y[2]) * np.exp(-1j * y[3])
        dy2dt = -1j * y[0] * np.conj(Omega) * np.exp(1j * y[3]) + 1j * np.conj(y[0]) * Omega * np.exp(-1j * y[3])
        dy3dt = -dy2dt
        dy4dt = E_gab

    else:
        raise IndexError("Dephasing must be either 0 or 1.")

    return np.array([dy1dt, dy2dt, dy3dt, dy4dt])
