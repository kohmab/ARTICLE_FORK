import matplotlib.pyplot as plt
import numpy as np
from Integrals import Coefficients
from Problem import Problem
from ClusterParameters import ClusterParameters


from scipy.special import spherical_jn


def rhs(r, w):
    return -G(k0, r)


multipoleIndex = 1
N = 1001
epsD = 1
r0 = 0.1
nu = 1
w = 1.5
k0 = 14


def G(k, r):
    return (
        spherical_jn(multipoleIndex, k * r) / k /
        spherical_jn(multipoleIndex, k, True)
    )


param = ClusterParameters(nu, r0, epsD)

p: Problem = Problem(param, N, multipoleIndex, w, rhsrho=rhs)

phi = p.getPhi()
rho = p.getRho()

r = p.getR()


eps = 1 - 1.0 / w / (w + 1j * nu)
kpa = np.sqrt(w * (w + 1j * nu) - 1) / r0
n = multipoleIndex
rho_teor = -(kpa**2) / k0**2 / (kpa**2 - k0**2) * (
    epsD * (n + 1) * (1 + epsD * (n + 1) * G(k0, 1))
    + (n + epsD * (n + 1)) * (k0**2 * r0**2 - epsD * (n + 1) * G(k0, 1))
) / (
    epsD * (n + 1) * (1 + epsD * (n + 1) * G(kpa, 1))
    + (n + epsD * (n + 1)) * (kpa**2 * r0**2 - epsD * (n + 1) * G(kpa, 1))
) * G(
    kpa, r
) + G(
    k0, r
) / (
    kpa**2 - k0**2
)
# phi_teor = C1*r + 4 * np.pi / kpa ** 2 * rho_teor


plt.figure(1)
plt.title(r"$\varphi$")
plt.plot(r, phi.real, "r")
plt.plot(r, phi.imag, "b")
# plt.plot(r, phi_teor.real, "k:")
# plt.plot(r, phi_teor.imag, "k:")
plt.grid()

plt.figure(2)
plt.title(r"$\rho$")
plt.plot(r, rho.real, "r")
plt.plot(r, rho.imag, "b")
plt.plot(r, rho_teor.real, "k:")
plt.plot(r, rho_teor.imag, "k:")
plt.grid()

plt.show()
