import numpy as np
from scipy.special import spherical_jn, jve
from ClusterParameters import ClusterParameters
from Oscillation import Oscillation


class DipoleOscillation(Oscillation):

    def __init__(self, N: int, parameters: ClusterParameters):
        super().__init__(N, 1, parameters)

    def __radFunc(self):
        return spherical_jn(1, self.__kp*self.r)/self.__kp/spherical_jn(1, self.__kp, True)

    def __updateFreq(self, w):
        self.freq = w
        if not self._isFreqChanged:
            return
        self.__kp = np.sqrt(w*(w+1j*self.nu) - 1. /
                            self.epsInf) / self.r0
        self.__eps = self.epsInf - 1. / w/(w + 1j * self.nu)

        v0 = jve(1.5, self.__kp)
        v1 = jve(2.5, self.__kp)
        self.__G = v0 / (v0 - v1*self.__kp)

        denum = self.__eps + 2 * self.epsD * \
            (1 + (self.__eps - self.epsInf)/self.epsInf*self.__G)
        self.__C = -3 * self.epsD/denum

        self._isFreqChanged = False

    def getPhi(self, w):
        self.__updateFreq(w)
        return self.__C*(self.r - self.__radFunc()/w/(w + 1j * self.nu)/self.epsInf)

    def getRho(self, w):
        self.__updateFreq(w)
        return -self.__C*self.__eps/(4.*np.pi*self.r0**2 * self.epsInf)*self.__radFunc()

    def getPsi(self, w):
        phi = self.getPhi(w)
        rho = self.getRho(w)
        return phi + 4*np.pi*self.r0**2 * rho
