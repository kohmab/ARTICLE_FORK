from typing import Callable
import numpy as np
from ClusterParameters import ClusterParameters
from TDMAsolver import TDMAsolver as Solver
from Integrals import Coefficients


class Problem:
    """
    @TODO
    Class for solving system of equations, determining\n
    the forced oscillations of the plasma sphere of unit radius.\n
    These equations are of form\n

        Δρ + kₚ²(ω)ρ = RHS(r,ω) Pₘ(cos θ),\n
        Δφ = -4πρ,\n

    and boundary conditions at r = 1 are\n

        ∂ᵣφ = -4π r₀² ∂ᵣρ = ε₀[ (2m+1)Q₀ - (m+1)φ ].\n

    It is assumed that the potential outside the sphere has the form\n

        φₑₓₜ = [Q₀ rᵐ + C / rᵐ⁺¹] Pₘ(cos θ).
    """

    __DIM: int = 2

    __multipoleNo: int

    __epsD: np.double

    __freq: np.double

    __nu: np.double

    __r0: np.double

    __kpaSq: np.complexfloating

    __Ac: np.ndarray

    __Bc: np.ndarray

    __Cc: np.ndarray

    __Av: np.ndarray

    __Bv: np.ndarray

    __Cv: np.ndarray

    __rhoextArr: np.ndarray

    __rhoextFunc: Callable[[np.ndarray, np.double], np.ndarray]

    __phiextArr: np.ndarray

    __phiextFunc: Callable[[np.ndarray, np.double], np.ndarray]

    __Q0: np.double

    __solver: Solver

    __coef: Coefficients

    __solved: bool

    __r: np.ndarray

    def _updateFreq(self):
        self.__solved = False

        self.__kpaSq = (
            self.__freq * (self.__freq + 1j * self.__nu) - 1./self.__epsInf
        ) / self.__r0**2

        self.__solver.A = self.__Ac + self.__Av * self.__kpaSq
        self.__solver.B = self.__Bc + self.__Bv * self.__kpaSq
        self.__solver.C = self.__Cc - self.__Cv * self.__kpaSq

        self.__rhoextArr = self.__rhoextFunc(self.__r, self.__freq)
        self.__phiextArr = self.__phiextFunc(self.__r, self.__freq)

        self.__solver.F = np.zeros((self.__coef.N, self.__DIM), dtype=complex)

        self.__solver.F[1:-1, 0] = -(
            self.__coef.alpha1[1:-1] * self.__rhoextArr[0:-2]
            + self.__coef.gamma1[1:-1] * self.__rhoextArr[1:-1]
            + self.__coef.beta1[1:-1] * self.__rhoextArr[2:]
        )

        self.__solver.F[1:-1, 0] += (
            + self.__coef.alpha2[1:-1] * self.__phiextArr[0:-2]
            - self.__coef.gamma2[1:-1] * self.__phiextArr[1:-1]
            + self.__coef.beta2[1:-1] * self.__phiextArr[2:]
        ) / 4. / np.pi

        self.__solver.F[1:-1, 0] += (
            - self.__coef.alpha0[1:-1] * self.__phiextArr[0:-2]
            - self.__coef.gamma0[1:-1] * self.__phiextArr[1:-1]
            - self.__coef.beta0[1:-1] * self.__phiextArr[2:]
        ) / 4. / np.pi * self.__multipoleNo * (self.__multipoleNo + 1)

        self.__solver.F[0, 0] = -(
            self.__coef.gamma1[0] * self.__rhoextArr[0]
            + self.__coef.beta1[0] * self.__rhoextArr[1]
        )

        self.__solver.F[0, 0] += (
            - self.__coef.gamma2[0] * self.__phiextArr[0]
            + self.__coef.beta2[0] * self.__phiextArr[1]
        ) / 4. / np.pi

        self.__solver.F[0, 0] += (
            - self.__coef.gamma0[0] * self.__phiextArr[0]
            - self.__coef.beta0[0] * self.__phiextArr[1]
        ) / 4. / np.pi * self.__multipoleNo * (self.__multipoleNo + 1)

        self.__solver.F[-1, 0] = -(
            self.__coef.gamma1[-1] * self.__rhoextArr[-1]
            + self.__coef.alpha1[-1] * self.__rhoextArr[-2]
        )

        self.__solver.F[-1, 0] += (
            + self.__coef.alpha2[-1] * self.__phiextArr[-2]
            - self.__coef.gamma2[-1] * self.__phiextArr[-1]
        ) / 4. / np.pi

        self.__solver.F[-1, 0] += (
            - self.__coef.alpha0[-1] * self.__phiextArr[-2]
            - self.__coef.gamma0[-1] * self.__phiextArr[-1]
        ) / 4. / np.pi * self.__multipoleNo * (self.__multipoleNo + 1)

        self.__solver.F[-1, 0] -= (
            self.__Q0
            * self.__epsD/self.__epsInf
            * (2 * self.__multipoleNo + 1)
            / (4 * np.pi * self.__r0**2)
        )

        self.__solver.F[-1, 1] += self.__Q0 * \
            self.__epsD/self.__epsInf * (2 * self.__multipoleNo + 1)

    def __zeroRHS(self, r: np.ndarray, w: np.double) -> np.ndarray:
        return np.zeros(self.__coef.N)

    def __init__(
        self,
        parameters: ClusterParameters,
        N: int,
        m: int,
        w: np.double,
        rhsrho: Callable[[np.ndarray, np.double], np.ndarray] = None,
        rhsphi: Callable[[np.ndarray, np.double], np.ndarray] = None,
        Q0: np.double = 0,
    ) -> None:
        """
        N    - Number of grid points\n
        m    - Multipole index\n
        nu   - Collisiton frequency [wp]\n
        w    - Frequency [wp]\n
        r0   - V0 / SphereRadius / wp  , \n
        epsD - Permittivity of external media (ε₀)\n
       @TODO rhs  - Rright hand side (source in the equation for rho). Function of r and freq\n
        Q0   - The coefficient before the component of the potential outside the sphere,\n
               which increases with increasing radius (φₑₓₜ ~ Q₀ rᵐ + C / rᵐ⁺¹)\n
        """
        self.__coef = Coefficients(N)
        self.__solver = Solver(N, self.__DIM, dtype=complex)
        self.__r = np.linspace(0, 1, N)

        self.__multipoleNo = m
        self.__epsD = parameters.epsD
        self.__epsInf = parameters.epsInf
        self.__nu = parameters.nu
        self.__freq = w
        self.__r0 = parameters.r0

        self.__rhoextFunc = rhsrho if rhsrho is not None else self.__zeroRHS
        self.__phiextFunc = rhsphi if rhsphi is not None else self.__zeroRHS
        self.__Q0 = Q0

        epsInf = parameters.epsInf
        epsD = parameters.epsD
        r0 = parameters.r0

        self.__Ac = np.zeros((N, self.__DIM, self.__DIM), dtype=complex)
        self.__Bc = np.zeros_like(self.__Ac, dtype=complex)
        self.__Cc = np.zeros_like(self.__Ac, dtype=complex)
        self.__Av = np.zeros_like(self.__Ac, dtype=complex)
        self.__Bv = np.zeros_like(self.__Ac, dtype=complex)
        self.__Cv = np.zeros_like(self.__Ac, dtype=complex)

        self.__Ac[:, 0, 0] = self.__coef.alpha2 - \
            m * (m + 1) * self.__coef.alpha0
        self.__Ac[:, 1, 1] = self.__Ac[:, 0, 0]
        self.__Ac[:, 1, 0] = 4 * np.pi / epsInf * self.__coef.alpha1
        self.__Av[:, 0, 0] = self.__coef.alpha1

        self.__Bc[:, 0, 0] = self.__coef.beta2 - \
            m * (m + 1) * self.__coef.beta0
        self.__Bc[:, 1, 1] = self.__Bc[:, 0, 0]
        self.__Bc[:, 1, 0] = 4 * np.pi / epsInf * self.__coef.beta1
        self.__Bv[:, 0, 0] = self.__coef.beta1

        self.__Cc[:, 0, 0] = self.__coef.gamma2 + \
            m * (m + 1) * self.__coef.gamma0
        self.__Cc[:, 1, 1] = self.__Cc[:, 0, 0]
        self.__Cc[:, 1, 0] = -4 * np.pi / epsInf * self.__coef.gamma1
        self.__Cc[-1, 0, 1] -= 1.0 / \
            (4 * np.pi * r0 * r0) * epsD/epsInf * (m + 1)
        self.__Cc[-1, 1, 1] += epsD/epsInf * (m + 1)
        self.__Cv[:, 0, 0] = self.__coef.gamma1

        self.__solved = False

    def setFreq(self, w: np.double) -> None:
        """
        Changes the frequency of the oscillation sourse
        """
        if self.__freq == w:
            return
        self.__freq = w
        self._updateFreq()

    def _solve(self) -> None:
        if self.__solved:
            return
        self._updateFreq()
        self.__solver.solve()
        self.__solved = True

    def getPhi(self) -> np.ndarray:
        """
        Returns the grid values of radial function Φ(r) determining\n
        the complex amplitude of the potential in sphere φ = Φ(r) Pₘ(cos θ).
        """
        self._solve()
        return self.__solver.solution[:, 1]

    def getRho(self) -> np.ndarray:
        """
        Returns the grid values of radial function R(r) determining\n
        the complex amplitude of the plasma density in sphere ρ = R(r) Pₘ(cos θ).
        """
        self._solve()
        return self.__solver.solution[:, 0]

    def getR(self) -> np.ndarray:
        """
        Returns the grid
        """
        return self.__r


if __name__ == "__main__":
    pass
