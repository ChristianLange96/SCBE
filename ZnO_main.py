import numpy as np
import multiprocessing as mp
from scipy.integrate import solve_ivp, odeint
from SCBE import SCBE_diff_eq
from energyband import energyband
from A_fun import A_fun
from F_fun import F_fun
from v_band import v_band
from scipy import signal
import matplotlib.pyplot as plt
from scipy.fft import fft


def nextpow2(x):
    """returns the smallest power of two that is greater than or equal to the
    absolute value of x.

    This function is useful for optimizing FFT operations, which are
    most efficient when sequence length is an exact power of two."""
    res = np.ceil(np.log2(x))
    return res.astype('int')  # we want integer values only but ceil gives float


def calc(pos, ky_n, kx_n, kx, ky, output):
    # Setup parameters (Copied from Mads)
    F_0     = 0.004
    omega   = 0.014
    T_0     = 2 * np.pi / omega
    n_c     = 10
    T_c     = T_0 * n_c / np.pi
    b_m     = 0     # Band model
    F_m     = 1     # Field model (1 = lin. pol. in x)

    # Lattice vector
    a       = np.array([5.32, 6.14, 9.38])    # This is from Prb 99 064302

    # D_vec
    d       = np.array([3.46, 3.46, 3.94])    # This is from Prb 99 064302 (assumed to be k-independent)


    # Creating k-grid OBS!! You should reconsider how this should be handled, maybe parse this from main instead.
    kx_n = kx_n  # Square grid for simplicity
    ky_n = ky_n
    # scale = 1
    # kx = np.linspace(-scale * np.pi, scale * np.pi, kx_n) / a[0]
    # ky = np.linspace(-scale * np.pi, scale * np.pi, ky_n) / a[1]
    d2k = 2 * np.pi / (a[0] * kx_n) * 2 * np.pi / (a[1] * kx_n)  # Infiniticimal area element in k-space

    # Starting conditions and time
    tmax = n_c * T_0
    tspan = np.linspace(0, tmax, 20000)
    y0 = np.array([0, 1.0 + 0j, 0, 0])  # Only the valence bond is populated


    # Setting up currents
    jx_ra = np.zeros(tspan.shape[0])
    jy_ra = np.zeros(tspan.shape[0])
    jx_er = np.zeros(tspan.shape[0])
    jy_er = np.zeros(tspan.shape[0])



    A = lambda t: A_fun(t, F_m, F_0)
    F = lambda t: F_fun(t, F_m, F_0)
    reltol = 1e-8
    abstol = 1e-10
    theta1 = np.pi / 3

    # Plugging into ODE
    for j in ky_n:
        ky_j = ky[j]
        for n in range(kx_n):
                k = np.array([kx[n], ky_j, 0])
                print(f"j,n = {j},{n} from process{pos}")
                dxy = np.array([d[0], d[1]])
                sol = solve_ivp(SCBE_diff_eq, [0, tmax], y0, t_eval=tspan, args=(0, T_0 / 4, k, F, A, d, b_m), rtol=reltol,
                               atol=abstol, first_step= 5.585848909541802e-04/2) #4.6551298233e-04)
                t = sol.t    # Unpacking t's
                y = sol.y.T  # Unpacking y's and making it a column vector

                vv = np.real(v_band(k + A(t), b_m, 0))  # v in valence band
                vc = np.real(v_band(k + A(t), b_m, 1))  # v in conduction band

                jx_ra = jx_ra + (y[:, 1] * vv[:, 0] + y[:,2] * vc[:, 0]) * d2k
                jy_ra = jy_ra + (y[:, 1] * vv[:, 1] + y[:,2] * vc[:, 1]) * d2k
                jx_er = jx_er + (dxy[0] * y[:, 0] * np.exp(1j * y[:, 3]) + dxy[0] * np.conj(y[:, 0]) * np.exp(-1j * y[:, 3])) * d2k
                jy_er = jy_er + (dxy[1] * y[:, 0] * np.exp(1j * y[:, 3]) + dxy[1] * np.conj(y[:, 0]) * np.exp(-1j * y[:, 3])) * d2k
    print(f"Hello! Am done with process{pos}")
    output.put((pos, {"jx_ra": jx_ra, "jy_ra": jy_ra, "jx_er": jx_er, "jy_er": jy_er}))
    # output.put((pos, jx_ra, jy_ra, jx_er, jy_er))
    # print(f"Really done {pos}")
    return 0


def main():
    if __name__ == '__main__':

        # Threads for processing. Used for splitting arrays. Should be the number of physical cores.
        # Library will take care of hyper threading.
        threads = 4

        # Lattice vector
        a = np.array([5.32, 6.14, 9.38])  # This is from Prb 99 064302

        # Creating k-grid
        n = 10
        kx_n = n  # Square grid for simplicity
        ky_n = n
        scale = 1
        kx = np.linspace(-scale * np.pi, scale * np.pi, kx_n) / a[0]
        ky = np.linspace(-scale * np.pi, scale * np.pi, ky_n) / a[1]
        d2k = 2 * np.pi / (a[0] * kx_n) * 2 * np.pi / (a[1] * ky_n)  # Infiniticimal area element in k-space

        # Splitting arrays for MP OBS!! For efficiency these should be split.
        # kxSplits = np.array_split(kx, threads)
        # kySplits = np.array_split(ky, threads)
        kxSplits = kx
        kySplits = ky
        # nSplits = [len(e) for e in kxSplits]
        nSplits = np.array_split(range(n), threads)
        print(nSplits)

        # Setup for MP
        output = mp.Queue()
        ky_n = 10
        processes = [mp.Process(target=calc, args=(pos, nSplits[pos], kx_n, kxSplits, kySplits, output)) for pos in range(threads)]

        # Doing calculations
        for p in processes:
            p.start()



        # Collecting results
        results = [output.get() for p in processes]

        for p in processes:
            p.join()

        results = sorted(results, key=lambda x: x[0])
        jx_ra = np.array([res[1]["jx_ra"] for res in results]).flatten().reshape(1, -1)
        jy_ra = np.array([res[1]["jy_ra"] for res in results]).flatten().reshape(1, -1)
        jx_er = np.array([res[1]["jx_er"] for res in results]).flatten().reshape(1, -1)
        jy_er = np.array([res[1]["jy_er"] for res in results]).flatten().reshape(1, -1)

        plotter(jx_ra, jy_ra, jx_er, jy_er)


def plotter(jx_ra, jy_ra, jx_er, jy_er):
    # Setup parameters (Copied from Mads)
    F_0 = 0.004
    omega = 0.014
    T_0 = 2 * np.pi / omega
    n_c = 10
    T_c = T_0 * n_c / np.pi
    b_m = 0  # Band model
    F_m = 1  # Field model (1 = lin. pol. in x)

    # Lattice vector
    a = np.array([5.32, 6.14, 9.38])  # This is from Prb 99 064302

    # D_vec
    d = np.array([3.46, 3.46, 3.94])  # This is from Prb 99 064302 (assumed to be k-independent)
    tmax = n_c * T_0
    tspan = np.linspace(0, tmax, 20000)
    dt = tspan[1] - tspan[0]
    jx_er = np.gradient(jx_er, dt)
    jy_er = np.gradient(jy_er, dt)

    # Applying window functions to the current
    window = signal.blackmanharris(jx_ra.shape[0])
    jx_ra *= window
    jy_ra *= window
    jx_er *= window
    jy_er *= window

    # Material properties (in terms of of fundamental freq.
    x_mingab = 8
    x_maxgab = 31
    x_specrange = 51

    # Axis of HHG-spectra
    L = tspan.shape[0]
    n = np.power(2, nextpow2(L))
    t_step = tspan[1] - tspan[0]
    f = np.arange(int(n / 2) + 1) / t_step * np.pi / (int(n / 2) + 1) / omega

    # Limits on plots
    xmin = 0
    xmax = 90
    ymin = 1e-35
    ymax = 1e-0

    # Calculating FFT of current - the HHG-spectra
    Y_ra = np.abs(fft(jx_ra, n)) ** 2 + np.abs(fft(jy_ra, n)) ** 2
    Y_er = np.abs(fft(jx_er, n)) ** 2 + np.abs(fft(jy_er, n)) ** 2

    fig, ax = plt.subplots(1)
    plt.semilogy(f, Y_ra[0: int(n / 2) + 1] / (n * n), label='Intraband', color='blue')
    plt.semilogy(f, Y_er[0: int(n / 2) + 1] / (n * n), label='Interband', color='red')
    plt.title("HHG-spectrum")
    plt.vlines(x_mingab, 0, 200, colors = 'k', linestyles='dashed', label='Min. bandgap')
    plt.vlines(x_maxgab, 0, 200, colors = 'k', linestyles='solid', label='Max. bandgap')
    plt.vlines(x_specrange, 0, 200, colors = 'k', linestyles='dotted', label='Spec. range')
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.xlabel(r'$\omega/\omega_0$')
    plt.ylabel(r'Intensity (arb. units)')
    leg = ax.legend()

    # Calculating total current
    jx_tot = jx_ra + jx_er
    jy_tot = jy_ra + jy_er
    J_tot = np.abs(fft(jx_tot, n)) ** 2 + np.abs(fft(jy_tot, n)) ** 2

    # Plotting total HHG-spectrum
    plt.figure(2)
    plt.semilogy(f, J_tot[0: int(n / 2) + 1] / (n * n), label ='Total Current')
    plt.title("HHG-spectrum")
    plt.vlines(x_mingab, 0, 200, colors = 'k', linestyles='dashed', label='Min. bandgap')
    plt.vlines(x_maxgab, 0, 200, colors = 'k', linestyles='solid', label='Max. bandgap')
    plt.vlines(x_specrange, 0, 200, colors = 'k', linestyles='dotted', label='Spec. range')
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.xlabel(r'$\omega/\omega_0$')
    plt.ylabel(r'Intensity (arb. units)')
    plt.legend()
    plt.show()
    print('Done')


if __name__ == '__main__':
    main()
