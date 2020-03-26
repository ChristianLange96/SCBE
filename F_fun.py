import numpy as np

def F_fun(t, type, F_0):

    if type == 1:  # Pol along x-direction
        F_0v = np.array([F_0, 0, 0])
        omega = 0.014               # Laser frequency
        T_0 = 2 * np.pi / omega     # Laser period
        n_c = 10                    # Number of cycles
        T_c = T_0 * n_c / np.pi     # Period of cycle
        ts = (np.sin(t / T_c) ** 2) * np.cos(omega * t) * F_0
        return np.array([ts, np.zeros(ts.shape), np.zeros(ts.shape)])

    elif type == 2:  # Pol along y-direction
        F_0v = np.array([0, F_0, 0])
        omega = 0.014               # Laser frequency
        T_0 = 2 * np.pi / omega     # Laser period
        n_c = 10                    # Number of cycles
        T_c = T_0 * n_c / np.pi     # Period of cycle
        return pow(np.sin(t / T_c),2) * pow(np.cos(omega * t),2) * F_0v
