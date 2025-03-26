from __future__ import annotations
import typing
import numpy as np

"""
Contains coefficients arising when solving the Helmholtz 
and Poisson equations inside a sphere of unit radius by the 
Galerkin method for test functions of form v = Λ_m LegendreP_n(cos (theta))   
"""


class Coefficients:
    __instances: typing.Dict[int, Coefficients] = {}

    __N: int
    """
        Number of grid nodes
    """

    __h: np.double
    """
        Grid step
    """

    @property
    def h(self):
        return self.__h

    @property
    def N(self):
        return self.__N

    __intRsqLL: np.ndarray
    """
        ∫r²Λₘ²dr 
    """

    __intRsqLLp1: np.ndarray
    """
        ∫r²ΛₘΛₘ₊₁dr 
    """

    __intRsqLLm1: np.ndarray
    """
        ∫r²ΛₘΛₘ₋₁dr 
    """

    __intLL: np.ndarray
    """
        ∫Λₘ²dr 
    """

    __intLLp1: np.ndarray
    """
        ∫ΛₘΛₘ₊₁dr 
    """

    __intLLm1: np.ndarray
    """
        ∫ΛₘΛₘ₋₁dr 
    """

    __intRsqdLdL: np.ndarray
    """
        ∫r²Λₘ'²dr 
    """

    __intRsqdLdLp1: np.ndarray
    """
        -∫r²Λₘ'Λₘ₊₁'dr 
    """

    __intRsqdLdLm1: np.ndarray
    """
        -∫r²Λₘ'Λₘ₋₁'dr 
    """

    def __new__(cls, *args, **kwargs):
        N: int = args[0]
        if N in cls.__instances.keys():
            return cls.__instances[N]
        instance = super().__new__(cls)
        cls.__instances[N] = instance
        return instance

    def __init__(self, N: int):
        if N < 3:
            raise RuntimeError("Grid must have at least three points.")

        h = 1.0 / (N - 1)

        self.__N = N

        self.__h = h

        r: np.ndarray = np.linspace(0, 1, N)

        # r^2 L_i L_j
        self.__intRsqLL = 2 * h * (r * r + h * h / 10) / 3
        self.__intRsqLL[0] = h**3 / 30
        self.__intRsqLL[-1] = h * (1 - h / 2 + h * h / 10) / 3

        self.__intRsqLLp1 = h * (r * r + h * r + 3 * h * h / 10) / 6
        self.__intRsqLLp1[-1] = 0

        self.__intRsqLLm1 = h * (r * r - h * r + 3 * h * h / 10) / 6
        self.__intRsqLLm1[0] = 0

        # r^2 dL_i dL_j
        self.__intRsqdLdL = 2 / h * (r * r + h * h / 3)
        self.__intRsqdLdL[0] = h / 3
        self.__intRsqdLdL[-1] = 1 / h * (1 - h + h * h / 3)

        self.__intRsqdLdLp1 = 1 / h * (r * r + h * r + h * h / 3)
        self.__intRsqdLdLp1[-1] = 0

        self.__intRsqdLdLm1 = 1 / h * (r * r - h * r + h * h / 3)
        self.__intRsqdLdLm1[0] = 0

        # L_i L_j
        self.__intLL = 2 * h / 3 * np.ones_like(r)
        self.__intLL[0] = h / 3
        self.__intLL[-1] = h / 3

        self.__intLLp1 = h / 6 * np.ones_like(r)
        self.__intLLp1[-1] = 0

        self.__intLLm1 = h / 6 * np.ones_like(r)
        self.__intLLm1[0] = 0

    @property
    def alpha0(self):
        """
        ∫ΛₘΛₘ₋₁dr
        """
        return self.__intLLm1

    @property
    def beta0(self):
        """
        ∫ΛₘΛₘ₊₁dr
        """
        return self.__intLLp1

    @property
    def gamma0(self):
        """
        ∫Λₘ²dr
        """
        return self.__intLL

    @property
    def alpha1(self):
        """
        ∫r²ΛₘΛₘ₋₁dr
        """
        return self.__intRsqLLm1

    @property
    def beta1(self):
        """
        ∫r²ΛₘΛₘ₊₁dr
        """
        return self.__intRsqLLp1

    @property
    def gamma1(self):
        """
        ∫r²Λₘ²dr
        """
        return self.__intRsqLL

    @property
    def alpha2(self):
        """
        -∫r²Λₘ'Λₘ₋₁'dr
        """
        return self.__intRsqdLdLm1

    @property
    def beta2(self):
        """
        -∫r²Λₘ'Λₘ₊₁'dr
        """
        return self.__intRsqdLdLp1

    @property
    def gamma2(self):
        """
        ∫r²Λₘ'²dr
        """
        return self.__intRsqdLdL


if __name__ == "__main__":
    pass
