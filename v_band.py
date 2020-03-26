import numpy as np

def v_band(k, n, m):
    # This function calculates the speed of charges in a specific band.
    # n refers to the model we want to use
    # m is the band in this model (0 is upper valence band, 1 is first cb and so on)
    # k is the vector at this point
    # Output is: Energy in this k-point at the m'th band in the n model.

    v_res = np.zeros((k.shape[0], 3))

    if n == 0:  # This is from Prb 99 064302

        # print(k[0])
        if m == 0:  # Valence band
            alphax = np.array([-0.0928, 0.0705, 0.0200, -0.0012, 0.0029, 0.0006])
            alphay = np.array([-0.0307, 0.0307])
            alphaz = np.array([-0.0059, 0.0059])
            a = np.array([5.32, 6.14, 9.38])
            # a(1) = 6.14;
            # a(2) = 5.32;

            v_res[:, 0] =  (-a[0] * (alphax[1] * np.sin(k[:,0] * a[0]) + 2 * alphax[2] *
            np.sin(2 * k[:,0]*a[0]) + 3 * alphax[3] * np.sin(3 * k[:,0] * a[0])+4 * alphax[4] *
            np.sin(4 * k[:,0]*a[0]) + 5 * alphax[5] * np.sin(5 * k[:,0] * a[0])))
            v_res[:, 1] = -a[1] * (alphay[1] * np.sin(k[:,1] * a[1]))
            v_res[:, 2] = -a[2] * (alphaz[1] * np.sin(k[:,2] * a[2]))

        if m == 1:  # Conduction band
            alphax = np.array([0.0898, -0.0814, -0.0024, -0.0048, -0.0003, -0.0009])
            alphay = np.array([0.1114, -0.1114])
            alphaz = np.array([0.0435, -0.0435])
            a = np.array([5.32, 6.14, 9.38])
            v_res[:, 0] = (-a[0] *(alphax[1] * np.sin(k[:,0] * a[0]) + 2 * alphax[2] *
            np.sin(2 * k[:,0]*a[0]) + 3 * alphax[3] * np.sin(3 * k[:,0] * a[0])+4 * alphax[4] *
            np.sin(4 * k[:,0]*a[0]) + 5 * alphax[5] * np.sin(5 * k[:,0] * a[0])))
            v_res[:, 1] = -a[1] * (alphay[1] * np.sin(k[:,1] * a[1]))
            v_res[:, 2] = -a[2] * (alphaz[1] * np.sin(k[:,2] * a[2]))

    if n == 1: # This is for ZnO according to PRB 99 014304 (taken from Mads)
        if m == 0: #Valence band
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
            kx = k[:,0]
            ky = k[:,1]
            kz = k[:,2]
            def f(kx,ky):
                return (2 * np.cos(np.sqrt(3) * ky * ay) + 4 * np.cos(np.sqrt(3) / 2 * ky * ay)
                        * np.cos(np.sqrt(3) * kx * ax))

            dfdx = lambda kx, ky :  -4 * np.sqrt(3) * ax * np.cos(np.sqrt(3) / 2 * ky * ay) * np.sin(np.sqrt(3) * kx * ax)
            dfdy = lambda kx, ky :  -2 * np.sqrt(3) * ay * (np.sin(np.sqrt(3) * ky * ay) + np.sin(np.sqrt(3) / 2 * ky * ay) * np.cos(np.sqrt(3) * kx * ax))

            # Band velocities
            v_res[:,0] = (t/2./np.sqrt(f(kx,ky)+q) * dfdx(kx,ky)+tp*dfdx(kx,ky))/u
            v_res[:,1] = (t/2./np.sqrt(f(kx,ky)+q) * dfdy(kx,ky)+tp*dfdy(kx,ky))/u
            v_res[:,2] = -a1z*az*np.sin(kz*az)

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
            kx = k[:, 0]
            ky = k[:, 1]
            kz = k[:, 2]

            def f(kx, ky):
                return (2 * np.cos(np.sqrt(3) * ky * ay) + 4 * np.cos(np.sqrt(3) / 2 * ky * ay)
                        * np.cos(np.sqrt(3) * kx * ax))
            dfdx = lambda kx, ky: -4 * np.sqrt(3) * ax * np.cos(np.sqrt(3) / 2 * ky * ay) * np.sin(np.sqrt(3) * kx * ax)
            dfdy = lambda kx, ky: -2 * np.sqrt(3) * ay * (np.sin(np.sqrt(3) * ky * ay) + np.sin(np.sqrt(3) / 2 * ky * ay) * np.cos(np.sqrt(3) * kx * ax))

            v_res[:, 0] = (t / 2. / np.sqrt(f(kx, ky) + q) * dfdx(kx, ky) + tp * dfdx(kx, ky)) / u
            v_res[:, 1] = (t / 2. / np.sqrt(f(kx, ky) + q) * dfdy(kx, ky) + tp * dfdy(kx, ky)) / u
            v_res[:, 2] = -a1z * az * np.sin(kz * az)

    return v_res