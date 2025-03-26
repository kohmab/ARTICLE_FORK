import numpy as np
import matplotlib.pyplot as plt


def AgEpsilon(hw):
    """
    Calculate the dielectric function of silver at a given wavelength.

    Parameters:
    hw (float or array): The wavelength(s) at which to calculate the dielectric function.

    Returns:
    AgEps (complex or array): The dielectric function of silver at the given wavelength(s).
    """

    # Silver data
    hw0 = np.array([0.64, 0.77, 0.89, 1.02, 1.14, 1.26, 1.39, 1.51, 1.64, 1.76, 1.88, 2.01, 2.13,
                    2.26, 2.38, 2.50, 2.63, 2.75, 2.88, 3.00, 3.12, 3.25, 3.37, 3.50, 3.62, 3.74,
                    3.87, 3.99, 4.12, 4.24, 4.36, 4.49, 4.61, 4.74, 4.86, 4.98, 5.11, 5.23, 5.36,
                    5.48, 5.60, 5.73, 5.85, 5.98, 6.10, 6.22, 6.35, 6.47, 6.60])

    n0 = np.array([0.24, 0.15, 0.13, 0.09, 0.04, 0.04, 0.04, 0.04, 0.03, 0.04, 0.05, 0.06, 0.05,
                   0.06, 0.05, 0.05, 0.05, 0.04, 0.04, 0.05, 0.05, 0.05, 0.07, 0.10, 0.14, 0.17,
                   0.81, 1.13, 1.34, 1.39, 1.41, 1.41, 1.38, 1.35, 1.33, 1.31, 1.30, 1.28, 1.28,
                   1.26, 1.25, 1.22, 1.20, 1.18, 1.15, 1.14, 1.12, 1.10, 1.07
                   ])

    k0 = np.array([14.08, 11.85, 10.10, 8.828, 7.795, 6.992, 6.312, 5.727, 5.242, 4.838, 4.483,
                   4.152, 3.858, 3.586, 3.324, 3.093, 2.869, 2.657, 2.462, 2.275, 2.070, 1.864,
                   1.657, 1.419, 1.142, 0.829, 0.392, 0.616, 0.964, 1.161, 1.264, 1.331, 1.372,
                   1.387, 1.393, 1.389, 1.378, 1.367, 1.357, 1.344, 1.342, 1.336, 1.325, 1.312,
                   1.296, 1.277, 1.255, 1.232, 1.212
                   ])

    Eps0 = (n0 + 1j * k0) ** 2

    AgEps = np.interp(hw, hw0, Eps0)

    return AgEps


if __name__ == '__main__':
    w = np.linspace(2, 6, 100)

    eps = AgEpsilon(w)

    fig, ax = plt.subplots()
    wp = 9
    ep_inf = 4
    nu = 1j*(w**2/ wp **2 - 1/(ep_inf - eps)) * wp / w

    ax.cla()
    ax.plot(w, eps.real, 'r', label='real')
    ax.plot(w, eps.imag, 'b', label='imag')
    ax.grid(True)
    axr = ax.twinx()
    axr.plot(w,nu.real,'k')
    axr.plot(w,nu.imag,'k--')
    axr.grid(True)

    plt.show()
