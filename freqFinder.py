import numpy as np
import scipy.special as ss
import scipy.optimize as so
import numba
from functools import cache
from sympy.functions.special.bessel import jn_zeros

from ClusterParameters import ClusterParameters


class FreqFinder:
    """
        Class for finding resonance frequecnies of 
        spherical metallic nanoparticle placed 
        in media with permittivity epsD
    """

    @staticmethod
    @numba.njit
    def __kp(eps, r0, epsInf):
        return np.sqrt(eps / (epsInf - eps) / epsInf + 0j) / r0

    @staticmethod
    @numba.njit
    def __eps(w, nu, epsInf):
        return epsInf - 1. / w / (w - 1j * nu)

    @staticmethod
    def __wFromEps(eps, nu, epsInf):
        return np.sqrt(1. / (epsInf - eps) - nu ** 2 / 4 + 0j) + 0.5j * nu

    @staticmethod
    def epsFromKp(kp, r0, epsInf):
        x = np.square(kp * r0) * epsInf
        return x / (1 + x) * epsInf

    @staticmethod
    def __zeroFunction(n, eps, r0, epsD, epsInf):
        kp = FreqFinder.__kp(eps, r0, epsInf)
        j = ss.jve(n + 0.5, kp)
        djkp = n * j - ss.jve(n + 1.5, kp) * kp
        return np.real((n * eps + epsD * (n + 1)) * djkp + epsD * n * (n + 1) * (eps - epsInf) / epsInf * j)

    @staticmethod
    def genGuessEps(n, N, r0, epsD, epsInf):
        result = np.zeros(N)
        result[0] = -epsD * (n + 1.) / n if n != 0 else np.nan
        if N == 1:
            return result
        vals = np.array(jn_zeros(n + 1, N - 1))
        result[1:] = FreqFinder.epsFromKp(vals, r0, epsInf)
        return result

    @staticmethod
    def genBetterGuessEpsForVps(n, N, r0, epsD, epsInf):
        result = np.zeros(N)
        x0 = np.array(jn_zeros(n + 1, N))
        C = (1 + n / (n + 1) * epsInf / epsD) * r0 ** 2 * epsInf
        kpa_guess = x0 + C / (1 / n - 2 * C) * x0
        return FreqFinder.epsFromKp(kpa_guess, r0, epsInf)

    def __init__(self, parameters: ClusterParameters, xtol=1e-9):
        """
            r0 -- charectiristic length of nonlocatity divided by the radius of the particle
            nu -- ratio of effective collision frequency and plasma frequency
            epsD -- permitivity of the surrounding media
        """

        if parameters.r0 > 1:
            raise Exception("Fermi radius is too big")
        self.__r0 = parameters.r0
        self.__nu = parameters.nu
        self.__epsD = parameters.epsD
        self.__epsInf = parameters.epsInf
        self.__xtol = xtol if xtol < 1e-9 else 1e-9

    def zeroFunc(self, n, eps):
        return FreqFinder.__zeroFunction(n, eps, self.__r0, self.__epsD, self.__epsInf)

    def getResonancePermittivities(self, n, Nz):
        """
            Returns the np.dnarray with values of nanoparticle dielectric function
            corresponding to the first Nz resonances of n-th multiplole mode of the nanopartilce.
            First element in array corresponds to the sufrace plasmon,
            subsequent ones correspond to the volume plasmons.
        """
        if Nz < 1:
            return np.array([], dtype=np.complex64)

        # Nz + 1 is need for np.isclose(spRootResult.root, 0) case
        guesses = FreqFinder.genGuessEps(
            n, Nz + 1, self.__r0, self.__epsD, self.__epsInf)

        if n == 0:
            return guesses[:-1]

        def F(x):
            return self.zeroFunc(n, x)

        def spFails():
            raise Exception(
                """Can not calculate resonance permittivity for surface plasmon""")

        result = np.zeros(Nz)

        # Surface plasmon:
        spGuess = guesses[0]
        spRootResult = so.root_scalar(F, x0=spGuess, xtol=self.__xtol)
        if not spRootResult.converged:
            spFails()

        if np.isclose(spRootResult.root, 0):
            spEpsInterval = np.linspace(0, guesses[1], 1000)
            zeroFuncValues = F(spEpsInterval)
            passThroughZero = zeroFuncValues[:-1] * zeroFuncValues[1:] < 0
            spGuess = spEpsInterval[np.where(passThroughZero)]
            spRootResult = so.root_scalar(F, x0=spGuess, xtol=self.__xtol)
            if not spRootResult.converged:
                spFails()

        result[0] = spRootResult.root

        # Volume plasmons
        for i, vpGuess in enumerate(guesses[1:-1]):
            vpRootResult = so.root_scalar(F, x0=vpGuess, xtol=self.__xtol)
            if not vpRootResult.converged:
                raise Exception(
                    f"""Can not calculate resonance permittivity for volume plasmon with No. {i + 1}""")

            result[i + 1] = vpRootResult.root
        return result

    def getResonanceFrequencies(self, n, Nz):
        """
            Returns the np.dnarray with first Nz eigenfrequencies 
            of n-th multiplole mode of the nanopartilce.
            First element in array corresponds to the sufrace plasmon,
            subsequent ones correspond to the volume plasmons.
        """
        resEps = self.getResonancePermittivities(n, Nz)
        return FreqFinder.__wFromEps(resEps, self.__nu, self.__epsInf)

    # def optFunc(self,  n, eps) :
    #     return FreqFinder.__abs2(self.zeroFunc(n, eps))

    @property
    def epsD(self):
        return self.__epsD

    @epsD.setter
    def epsD(self, epsD):
        self.__epsD = epsD


if __name__ == "__main__":
    from ClusterParameters import ClusterParameters
    import matplotlib.pyplot as plt

    nu = 0.001
    r0 = .01

    epsD = 3
    epsInf = 6
    params = ClusterParameters(nu, r0, epsD, epsInf)
    ff = FreqFinder(params)
    n = 1
    eps = np.linspace(-2, epsInf, 100000)
    Nz = 50
    zf = ff.zeroFunc(n, eps)
    zeros0 = FreqFinder.genGuessEps(n, Nz, r0, epsD, epsInf)
    zeros1 = FreqFinder.genBetterGuessEpsForVps(n, Nz, r0, epsD, epsInf)
    zeros = ff.getResonancePermittivities(n, Nz)
    print(ff.getResonanceFrequencies(n, Nz))
    # def F(eps): return ff.optFunc(n, eps)
    # res = minimize(F, x0=-2)
    # print(res)
    fig, ax = plt.subplots()
    ax.plot(eps, zf, 'r')
    ax.scatter(zeros0, np.zeros(Nz), c="black")
    ax.scatter(zeros1, np.zeros(Nz), c="green")
    ax.scatter(zeros, np.zeros_like(zeros), c="red")
    ax.grid()
    plt.ylim([-1, 1])
    plt.show()
    # ax.plot(eps, of, 'k')
