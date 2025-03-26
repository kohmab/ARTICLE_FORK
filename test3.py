from ClusterParameters import ClusterParameters
from DipoleOscillation import DipoleOscillation
from NonlinearSources import NonlinearSources
import numpy as np
import matplotlib.pyplot as plt

from Problem import Problem

if __name__ == "__main__":
    nu = 0.01
    r0 = 0.1
    epsD = 1.
    epsInf = 1.
    beta = 0.01
    N = 10001
    w = 0.5

    params = ClusterParameters(nu, r0, epsD=epsD, epsInf=epsInf)
    ns = NonlinearSources(params, beta)

    p = Problem(
        params, N, 0, w, rhsrho=ns.rhoFunctions[0], rhsphi=ns.phiFunctions[0])

    dip = DipoleOscillation(N, params)

    rho = p.getRho()
    phi = p.getPhi()
    r = p.getR()

    plt.figure(1)
    plt.title(r'$\rho$')
    plt.plot(r, rho.real, 'r')
    plt.plot(r, rho.imag, 'b')
    plt.grid()

    plt.figure(2)
    plt.title(r'$\varphi$')
    plt.plot(r, phi.real, 'r')
    plt.plot(r, phi.imag, 'b')
    plt.grid()

    plt.figure(3)
    plt.title(r'func')
    kp = ns._kp(w)
    fun = ns.rhoFunctions[0](r, w)
    plt.plot(r, fun.real, 'r')
    plt.plot(r, fun.imag, 'b')
    plt.grid()

    plt.figure(4)
    plt.title(r'dip')
    kp = ns._kp(w)
    fun = dip.getPsi(w)
    plt.plot(r, fun.real, 'r')
    plt.plot(r, fun.imag, 'b')
    plt.grid()

    plt.show()
