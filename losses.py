from typing import List
from ClusterParameters import ClusterParameters
from DipoleOscillation import DipoleOscillation
from Oscillation import Oscillation
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os.path
import pickle

from SecHarmOscillation import SecHarmOscillation
from freqFinder import FreqFinder

# wp = 5.71 / 6.6e-16
# nu = 0.0276 / 5.71
# Vf = 1.07e8
# epsInf = 1
# a = 4e-7
#
# epsD = 1.25
# beta = .02

wp = 5 / 6.6e-16
nu = 0.02
Vf = 1.5e8
epsInf = 2
a = 4e-7

epsD = 8.24
beta = .1

N = 500

wmin = 0.02
wmax = 1
Nw = 1500

def getLossesAtOneFreq(osc: Oscillation, w: float):
    psi = osc.getPsi(w)
    rhoConj = osc.getRho(w).conj()
    rsq = np.square(osc.r)
    # @TODO calculate with Integrals
    integralOverR = np.trapz(psi*rhoConj*rsq, osc.r)
    integralOverTheta = 2./(osc.multipoleN0*2 + 1)
    integralOverPhi = 2*np.pi
    coef = osc.nu*osc.freq/(osc.freq + 1j*osc.nu)/2
    return np.real(coef*integralOverPhi*integralOverTheta*integralOverR)


def getLosses(osc: Oscillation, w: np.ndarray) -> np.ndarray:
    result = np.zeros_like(w)
    print(f"Calculating losses for multipole #{osc.multipoleN0}...")
    total = w.size
    for i, freq in tqdm(enumerate(w), total=total):
        result[i] = getLossesAtOneFreq(osc, freq)
    return result


def generateFilename(*args) -> str:
    res = 0
    for arg in args:
        res += res*31 + hash(arg)
    return hex(res)


if __name__ == "__main__":

    useCache = True

    # # Na
    # wp = 5 / 6.6e-16
    # nu = 0.02
    # Vf = 1.4e8
    # epsInf = 3
    # a = 3.5e-7

    V0 = np.sqrt(3 / 5) * Vf
    r0 = V0 / wp
    alpha = r0 / a


    # wp = 9 / 6.6e-16
    # nu = 0.02
    # Vf = 2.0e8
    # V0 = np.sqrt(3 / 5) * Vf
    # r0 = V0 / wp
    # epsInf = 3
    # a = 5e-7
    # alpha = r0 / a
    # epsD = 5.4
    # beta = .1

    w = np.linspace(wmin, wmax, Nw)

    params = ClusterParameters(nu, alpha, epsD, epsInf)
    params_dip = ClusterParameters(nu + 3/4*Vf/a/wp, alpha, epsD, epsInf)
    oscillations: List[Oscillation] = []
    oscillations.append(SecHarmOscillation(N, 0, params, beta))
    oscillations.append(DipoleOscillation(N, params_dip))
    oscillations.append(SecHarmOscillation(N, 2, params, beta))

    filename = "./savedResults/" + \
        generateFilename(nu, alpha, epsD, epsInf, beta, N, Nw, wmin, wmax)
    if os.path.isfile(filename) and useCache:
        print("Loading saved results...")
        with open(filename, "rb") as handle:
            w, losses = pickle.load(handle)
    else:
        losses: List[np.ndarray] = []
        for osc in oscillations:
            losses.append(getLosses(osc, w))
        with open(filename, "wb") as handle:
            pickle.dump((w, losses), handle)

    colors = ["g", "b", "r"]
    fig = plt.figure(1)
    ff = FreqFinder(params)
    for i, l in enumerate(losses):
        plt.plot(w, l, colors[i])
        resFreq = ff.getResonanceFrequencies(
            oscillations[i].multipoleN0, 10).real
        if isinstance(oscillations[i], SecHarmOscillation):
            resFreq /= 2.
        plt.scatter(resFreq, np.zeros(10), c=colors[i], marker="x")
    plt.plot(w, sum(losses), 'y')
    plt.xlim((wmin, wmax))
    plt.grid()
    text = f"bt={beta}_wp={wp*6.6e-16}ev_de={nu}_epsInf={epsInf}_Vf={Vf/1e8}(1e8cms)_a={a*1e7}nm_epsD={epsD}"
    plt.title(text)
    filename = f"./loss_res_curves/" + text + ".png"
    fig.savefig(filename)
    plt.show()
