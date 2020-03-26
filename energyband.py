import numpy as np


def energyband(k, n, m):
    # n refers to the model we want to use
    # m is the band in this model (0 is upper valence band, 1 is first cb and so on)
    # k is the vector at this point

    # Output is: Energy in this k-point at the m'th band in the n model.

    if n == 0:  # This is from Prb 99 064302

        E_band = np.zeros(k.shape[0])
        # print(k[0])
        if m == 0:  # Valence band
            alphax = np.array([-0.0928, 0.0705, 0.0200, -0.0012, 0.0029, 0.0006])
            alphay = np.array([-0.0307, 0.0307])
            alphaz = np.array([-0.0059, 0.0059])
            a = np.array([5.32, 6.14, 9.38])

            E_band[0] = (alphax[0] + alphax[1] * np.cos(k[0] * a[0]) + alphax[2] *
                         np.cos(2 * k[0] * a[0]) + alphax[3] * np.cos(3 * k[0] * a[0]) + alphax[4] *
                         np.cos(4 * k[0] * a[0]) + alphax[5] * np.cos(5 * k[0] * a[0]))
            E_band[1] = alphay[0] + alphay[1] * np.cos(k[1] * a[1])
            E_band[2] = alphaz[0] + alphaz[1] * np.cos(k[2] * a[2])
            E_band = np.sum(E_band)


        elif m == 1:  # Conduction band
            alphax = np.array([0.0898, -0.0814, -0.0024, -0.0048, -0.0003, -0.0009])
            alphay = np.array([0.1114, -0.1114])
            alphaz = np.array([0.0435, -0.0435])
            a = np.array([5.32, 6.14, 9.38])
            E_band[0] = (0.1213 + alphax[0] + alphax[1] * np.cos(k[0] * a[0]) + alphax[2] *
                         np.cos(2 * k[0] * a[0]) + alphax[3] * np.cos(3 * k[0] * a[0]) + alphax[4] *
                         np.cos(4 * k[0] * a[0]) + alphax[5] * np.cos(5 * k[0] * a[0]))
            E_band[1] = alphay[0] + alphay[1] * np.cos(k[1] * a[1])
            E_band[2] = alphaz[0] + alphaz[1] * np.cos(k[2] * a[2])
            E_band = np.sum(E_band)

    elif n == 1:  # This is for ZnO according to PRB 99 014304 (taken from Mads)
        if m == 0:  # Valence band
            # Constants in atomic units
            t = 2.38
            tp = -0.0020
            u = 27.1
            p = -7.406
            q = 4.0
            a0z = -0.0059
            a1z = 0.0059
            ax = 5.32
            ay = 6.14
            az = 9.83

            # Geometry
            kx = k[0]
            ky = k[1]
            kz = k[2]

            def f(kx, ky):
                return (2 * np.cos(np.sqrt(3) * ky * ay) + 4 * np.cos(np.sqrt(3) / 2 * ky * ay)
                        * np.cos(np.sqrt(3) * kx * ax))

            # Energy
            E_bandxy = (t * np.sqrt(f(kx, ky) + q) + tp * f(kx, ky) + p) / u
            E_bandz = a0z + a1z * np.cos(kz * az)
            E_band = E_bandxy


        elif m == 1:  # Conduction band
            t = -2.38
            tp = -0.0020
            u = 27.1
            p = 10.670
            q = 3.3
            a0z = -0.0435
            a1z = 0.0435
            ax = 5.32
            ay = 6.14
            az = 9.83

            # Geometry
            kx = k[0]
            ky = k[1]
            kz = k[2]

            def f(kx, ky):
                return (2 * np.cos(np.sqrt(3) * ky * ay) + 4 * np.cos(np.sqrt(3) / 2 * ky * ay)
                        * np.cos(np.sqrt(3) * kx * ax))

            # Energy
            E_bandxy = (t * np.sqrt(f(kx, ky) + q) + tp * f(kx, ky) + p) / u
            E_bandz = a0z + a1z * np.cos(kz * az)
            E_band = E_bandxy

    return E_band  ## Why do we sum with 2 here?
