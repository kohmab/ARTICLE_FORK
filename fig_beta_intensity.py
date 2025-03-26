import numpy as np
import matplotlib.pyplot as plt

_c = 3e10
_m = 0.91e-27
_e = 4.8e-10
_V0 = np.sqrt(3 / 5) * 1e8


def E0(intensity):
    return np.sqrt(intensity * 1e7 * 8 * np.pi / _c)


def beta_div_a(intensity):  # 1/nm
    E = E0(intensity)
    return 1e-7 * _e * E / _m / (_V0 ** 2)


def calc_beta(intensity, V0, a):
    """
    @param a  - cluster radius [nm]
    @param V0 - speed of first sound [cm / s]
    @param intensity - incident field intensity [W / cm^2]
    """
    corr = (_V0 / V0) ** 2 * 1e7 * a
    return beta_div_a(intensity) * corr


if __name__ == '__main__':
    I = 1e8 * 10 ** (np.linspace(0, 2, 100))
    plt.figure()
    plt.semilogx(I, beta_div_a(I))
    plt.grid(True)
    plt.show()
