from scipy.special import spherical_jn
import matplotlib.pyplot as plt
import numpy as np
from ClusterParameters import ClusterParameters
from Problem import Problem


def rhs(r, w):
    return np.sinh(r * 10)


multipoleIndex = 1
N = 501
epsD = 1
r0 = 0.1
nu = 0.1
w = 1.4

param = ClusterParameters(nu, r0, epsD)

p: Problem = Problem(param, N, multipoleIndex, w, Q0=-1)

phi = p.getPhi()
rho = p.getRho()

r = p.getR()


eps = 1 - 1.0 / w / (w + 1j * nu)
kpa = np.sqrt(w * (w + 1j * nu) - 1) / r0


def G(r): return spherical_jn(1, r * kpa) / spherical_jn(1, kpa, True) / kpa


C1 = -3 * epsD / (eps + 2 * epsD * (1 + (eps - 1) * G(1)))
rho_teor = -eps / (4 * np.pi * r0**2) * C1 * G(r)
# phi_teor = C1 * r + 4 * np.pi / kpa**2 * rho_teor
phi_teor = C1 * (r - 1/w/(w+1j*nu)*G(r))


plt.figure(1)
plt.title(r"$\varphi$")
plt.plot(r, phi.real, "r")
plt.plot(r, phi.imag, "b")
plt.plot(r, phi_teor.real, "k:")
plt.plot(r, phi_teor.imag, "k:")
plt.grid()

plt.figure(2)
plt.title(r"$\rho$")
plt.plot(r, rho.real, "r")
plt.plot(r, rho.imag, "b")
plt.plot(r, rho_teor.real, "k:")
plt.plot(r, rho_teor.imag, "k:")
plt.grid()

plt.show()
