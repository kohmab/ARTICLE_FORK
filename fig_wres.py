import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
from tqdm import tqdm

from ClusterParameters import ClusterParameters
from freqFinder import FreqFinder
from losses import wp, a, nu, Vf

V0 = np.sqrt(3 / 5) * Vf
r0 = V0 / wp
alpha = r0 / a

Nres = 7
NepsD = 50
epsD = np.linspace(1, 9, NepsD)

colors = {0: '#77AB30', 1: '#000000', 2: '#D95A19'}
symbols = {0: 'o', 2: 'x'}
liness = {0: '-', 1: '--'}


def res_ind_labels(x, y, epsInf, m, nres):
    # text_shifts = defaultdict(lambda: (0, 0))
    # text_shifts[(4, 0, 1)] = (0.03, 0.017)
    # text_shifts[(4, 0, 2)] = (0.125, 0.017)
    # text_shifts[(4, 0, 3)] = (0.17, 0.019)
    #
    # text_shifts[(4, 2, 1)] = (-0.1, -0.018)
    # text_shifts[(4, 2, 2)] = (-0.12, -0.018)
    #
    # text_shifts[(2, 0, 1)] = (0.15, 0.012)
    # text_shifts[(2, 0, 2)] = (0.125, 0.017)
    # text_shifts[(2, 0, 3)] = (0.17, 0.019)
    #
    # text_shifts[(2, 2, 1)] = (-0.1, -0.018)
    # text_shifts[(2, 2, 2)] = (-0.12, -0.018)
    text_pos = {0: {"ha": "left", "va": "bottom"}, 2: {"ha": "right", "va": "top"}}
    text_shift = {0: (0.00, 0.007), 2: (0.0, -0.01)}
    dx, dy = text_shift[m]
    return x + dx, y + dy, f"{nres}", text_pos[m]


fig, host = plt.subplots(figsize=(5, 4), layout='constrained')

for ei, epsInf in enumerate((2, 4)):
    par = ClusterParameters(nu, alpha, 1, epsInf)
    ff = FreqFinder(parameters=par)

    limw = np.array([0.2, 0.6])

    resFreq = np.zeros((NepsD, 3, Nres))

    for i, ed in tqdm(enumerate(epsD), total=NepsD):
        ff.epsD = ed
        for j in range(3):
            resFreq[i, j, :] = np.real(ff.getResonanceFrequencies(n=j, Nz=Nres))

    # resFreq *= np.sqrt(epsInf)
    freq_dip_doubled = resFreq[:, 1, 0] * 2

    rng = [freq_dip_doubled.min(), freq_dip_doubled.max()]
    points = np.zeros((3, Nres, 2))
    interp = interp1d(freq_dip_doubled, epsD, kind='cubic')
    for m in (0, 2):
        for r in range(Nres - 1):
            freq = resFreq[0, m, r + 1]
            epsDres = None
            if freq > rng[0] and freq < rng[1]:
                epsDres = interp(freq)
                host.text(*res_ind_labels(epsDres, freq, epsInf, m, r + 1), size=8, weight="bold", color=colors[m])
                points[m, r, 0] = freq
                points[m, r, 1] = epsDres

            sc = host.scatter(points[m, :, 1], points[m, :, 0], color=colors[m], marker=symbols[m])
            pl = host.plot(epsD, freq_dip_doubled, color=colors[1])
            sc.set_zorder(10)
            # pl.set_zorder(0)
        #
        # for m in (0, 2):
        #     for i in range(1, Nres):
        #         freqs = resFreq[:, m, i]
        #         host.plot(epsD, freqs, color=colors[m], linestyle=liness[ei])

    # ax1.grid(True)
host.text(4.8, 0.667, r"$\varepsilon_{\infty} = 2$", rotation=-25, size=10, ha="center", va="bottom")
host.text(4.7, 0.604, r"$\varepsilon_{\infty} = 4$", rotation=-23.5, size=10, ha="center", va="bottom")

host.grid(True)
host.set_ylim((0.51, 1.05))
host.set_xlim((1, 8.5))
host.set_ylabel(r"$2\omega^{(1,0)}/\omega_p$")
host.set_xlabel(r"$\varepsilon_d$")
# ax1.set_ylim(2 * limw)

plt.show()
