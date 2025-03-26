import numpy as np


class ClusterParameters:
    def __init__(self,
                 nu: np.double, r0: np.double,
                 epsD: np.double = 1.,
                 epsInf: np.double = 1.) -> None:
        self.__nu = nu
        self.__r0 = r0
        self.__epsD = epsD
        self.__epsInf = epsInf

    @property
    def nu(self):
        return self.__nu

    @property
    def r0(self):
        return self.__r0

    @property
    def epsD(self):
        return self.__epsD

    @property
    def epsInf(self):
        return self.__epsInf
