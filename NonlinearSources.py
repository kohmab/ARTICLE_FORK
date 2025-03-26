
from ClusterParameters import ClusterParameters
import numpy as np
from scipy.special import spherical_jn


class NonlinearSources():
    __instances = {}

    def __new__(cls, *args, **kwargs):
        if args in cls.__instances.keys():
            return cls.__instances[args]
        instance = super().__new__(cls)
        cls.__instances[args] = instance
        return instance

    def __init__(self, parameters: ClusterParameters, beta=0.) -> None:
        self._parameters = parameters
        self._beta = beta
        self._rhoFunctions = {0: self._rhoExtMono,
                              2: self._rhoExtQuad}
        self._phiFunctions = {0: self._phiExtMono,
                              2: self._phiExtQuad}

    @property
    def rhoFunctions(self):
        return self._rhoFunctions

    @property
    def phiFunctions(self):
        return self._phiFunctions

    def _memoize(function):
        prevScalArg = np.inf
        prevVecArgSize = np.inf
        savedValue = None

        def wrapper(self, vector, scalar):
            nonlocal prevScalArg, prevVecArgSize, savedValue
            if scalar == prevScalArg and prevVecArgSize == vector.size:
                return savedValue
            else:
                prevScalArg = scalar
                prevVecArgSize = vector.size
                value = function(self, vector, scalar)
                savedValue = value
                return value
        return wrapper

    @property
    def beta(self):
        return self._beta

    @property
    def parameters(self):
        return self._parameters

    def _kp(self, w):
        return np.sqrt(w*(w+1j*self.parameters.nu) - 1. /
                       self.parameters.epsInf) / self.parameters.r0

    def _eps(self, w):
        return self.parameters.epsInf - 1./w/(w+1.j * self.parameters.nu)

    def _G(self, kp, r):
        return spherical_jn(1, kp*r)/kp/spherical_jn(1, kp, True)

    def C(self, w):
        kp = self._kp(w)
        eps = self._eps(w)
        epsInf = self.parameters.epsInf
        epsD = self.parameters.epsD
        val = 1 + (eps - epsInf) / epsInf*self._G(kp, 1)
        return -3.*self.parameters.epsD / (eps + 2*epsD*val)

    @_memoize
    def _f1(self, r, kp):
        return np.square(spherical_jn(1, kp*r)/spherical_jn(1, kp, True))

    @_memoize
    def _f2(self, r, kp):
        return spherical_jn(1, kp*(r+1e-16), True)/spherical_jn(1, kp, True)

    @_memoize
    def _f3(self, r, kp):
        return spherical_jn(1, kp*(r+1e-16))/spherical_jn(1, kp, True)/kp/(r+1e-16)

    @_memoize
    def _cosSqPartPhi(self, r, w):
        kp = self._kp(w)
        eps = self._eps(w)
        epsInf = self.parameters.epsInf
        nu = self.parameters.nu
        return self.beta*self.C(w)**2/4 * (
            eps/epsInf/w/(w+1.j*nu) * self._f1(r, kp) -
            1/(w+1.j*nu)**2*(1-self._f2(r, kp))**2
        )

    @_memoize
    def _sinSqPartPhi(self, r, w):
        kp = self._kp(w)
        eps = self._eps(w)
        epsInf = self.parameters.epsInf
        nu = self.parameters.nu
        return -self.beta*self.C(w)**2/4 * (1-self._f3(r, kp))**2/(w+1.j*nu)**2

    @_memoize
    def _cosSqPartRho(self, r, w):
        kp = self._kp(w)
        eps = self._eps(w)
        epsInf = self.parameters.epsInf
        nu = self.parameters.nu
        r0 = self.parameters.r0
        val = self.beta*self.C(w)**2 / (4.*np.pi*r0**2) * \
            (2.j*w-nu) / 2. / (1.j*w-nu) * \
            eps / epsInf
        return val * (self._f1(r, kp) + self._f2(r, kp)*(1-self._f2(r, kp)))

    @_memoize
    def _sinSqPartRho(self, r, w):
        kp = self._kp(w)
        eps = self._eps(w)
        epsInf = self.parameters.epsInf
        nu = self.parameters.nu
        r0 = self.parameters.r0
        val = self.beta*self.C(w)**2 / (4.*np.pi*r0**2) * \
            (2.j*w-nu) / 2. / (1.j*w-nu) * \
            eps / epsInf
        return + val * self._f3(r, kp)*(1-self._f3(r, kp))

    def _rhoExtQuad(self, r, wDoubled):
        w = wDoubled / 2
        return (self._cosSqPartRho(r, w)-self._sinSqPartRho(r, w))*2./3.

    def _phiExtQuad(self, r, wDoubled):
        w = wDoubled / 2
        return (self._cosSqPartPhi(r, w)-self._sinSqPartPhi(r, w))*2./3.

    def _rhoExtMono(self, r, wDoubled):
        w = wDoubled / 2
        return (self._cosSqPartRho(r, w)+2*self._sinSqPartRho(r, w))/3.

    def _phiExtMono(self, r, wDoubled):
        w = wDoubled / 2
        return (self._cosSqPartPhi(r, w)+2*self._sinSqPartPhi(r, w))/3.


if __name__ == "__main__":
    pass
