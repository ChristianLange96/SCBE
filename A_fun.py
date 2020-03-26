import numpy as np


def A_fun(t,type, F_0):
    if type == 1: # Lin. pol in x-direction, taken from Mads
        F_0v = np.array([F_0, 0, 0])
        omega = 0.014
        T_0 = 2 * np.pi / omega
        n_c = 10
        T_c = T_0 * n_c / np.pi
        ts = (1 / 4 * (np.sin(t * (omega - 2 * 1 / T_c)) / (omega - 2 * 1 / T_c)
                    + np.sin(t * (omega + 2 * 1 / T_c)) / (omega + 2 * 1 / T_c) - 2 * np.sin(omega * t) / omega)) *F_0

        return np.array([ts, np.zeros(ts.shape), np.zeros(ts.shape)]).T



    if type == 2: # Lin. pol in y-direction, taken from Mads
        F_0v = np.array([0, F_0, 0])
        omega = 0.014
        T_0 = 2 * np.pi / omega
        n_c = 10
        T_c = T_0 * n_c / np.pi
        return (1 / 4 * (np.sin(t * (omega - 2 * 1 / T_c)) / (omega - 2 * 1 / T_c)
                     + np.sin(t * (omega + 2 * 1 / T_c)) / (omega + 2 * 1 / T_c) - 2 * np.sin(omega * t) / omega) * F_0v)
