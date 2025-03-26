from numpy import pi
from ClusterParameters import ClusterParameters
from NonlinearSources import NonlinearSources
from Oscillation import Oscillation
from Problem import Problem


class SecHarmOscillation(Oscillation):

    def __init__(self, N: int, m: int, parameters: ClusterParameters, beta: float) -> None:
        if m != 0 and m != 2:
            raise RuntimeError("Invalid multipole index")
        super().__init__(N, m, parameters)
        self.__ns = NonlinearSources(parameters, beta)
        self.__problem = Problem(
            parameters, N, m, None, self.__ns.rhoFunctions[m], self.__ns.phiFunctions[m])

    def getPhi(self, w):
        self.__problem.setFreq(2*w)
        self.freq = 2*w
        phi = self.__problem.getPhi()
        return phi

    def getRho(self, w):
        self.__problem.setFreq(2*w)
        self.freq = 2*w
        rho = self.__problem.getRho()
        return rho

    def getPsi(self, w):
        rho = self.getRho(w)
        phi = self.getPhi(w)
        phiExt = self.__ns.phiFunctions[self.multipoleN0](self.r, self.freq)
        return phi + (phiExt + 4*pi*rho)*self.r0**2


if __name__ == "__main__":
    pass
    # params = ClusterParameters(0.1, 0.05, 1, 1)
    # N = 1001
    # beta = 0.01
    # osc = SecHarmOscillation(N, 0, params, beta)
    # psi = osc.getPsi(1.5)
    # import matplotlib.pyplot as plt

    # plt.figure(1)

    # plt.plot(osc.r, psi.real, 'r')
    # plt.plot(osc.r, psi.imag, 'b')
    # plt.grid()
    # plt.show()
