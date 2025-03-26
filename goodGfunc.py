from scipy.special import spherical_jn, jve
import numpy as np
import matplotlib.pyplot as plt


def Gnaive(n, kp):
    return spherical_jn(n, kp) / kp / spherical_jn(n, kp, True)


def Gsmart(n, kp):
    v0 = jve(n + 0.5, kp)
    v1 = jve(n + 1.5, kp)

    return v0 / (n * v0 - v1*kp)


r0 = 0.001
nu = 0.001
w = np.linspace(0, 2, 100000)
kp = np.sqrt(w*(w + 1j*nu) - 1.)/r0
n = 1

naive = Gnaive(n, kp)
smart = Gsmart(n, kp)

plt.figure(1)
plt.plot(w, naive.real, 'c')
plt.plot(w, naive.imag, 'm')
plt.plot(w, smart.real, 'r--')
plt.plot(w, smart.imag, 'b--')
plt.grid()
plt.show()
