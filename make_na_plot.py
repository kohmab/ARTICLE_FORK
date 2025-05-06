import shelve
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np

from ClusterParameters import ClusterParameters as ClusPar
from fig_beta_intensity import calc_beta
from freqFinder import FreqFinder as FrFnd
from losses import generateFilename


# Sodium cluster and field:
Vf = 1.07e8  # cm/s
wp_eV = 5.71  # eV
nu_eV = 0.03  # 0.0276  # eV
epsInf = 1
# field_intensity = 1e8  # W/cm^2

# Program parameters
epsDmin = 1
epsDmax = 1.8
NepsD = 400
Nw = 151
Nr = 100
hash_str = []
#radii = (7e-7, 10e-7)
radii = (5e-7, 7e-7, 10e-7)
field_intensitys = (0.5e8, 1e8, 1e8)
for radius in radii:  # cm
    field_intensity = field_intensitys[radii.index(radius)]
    sorted_locals = sorted(locals().items(), key=lambda item: item[0])
    hash_names = {k: v for k, v in sorted_locals if isinstance(v, float) or isinstance(v, int)}
    hash_str.append(generateFilename(*hash_names.values()))
    #
fig, ax = plt.subplots(figsize=(5, 4), layout='constrained')

colors = {0: '#77AB30', 1: '#D95A19', 2:'#0b62a4'}
linestyles = ["--", ":", "-."]

for i, radius in enumerate(radii):
    # Derived parameters
    V0 = np.sqrt(3 / 5) * Vf
    wp = wp_eV / 6.6e-16
    r0 = V0 / wp
    nu = nu_eV / wp_eV
    nu_dip = nu + 3 / 4 * Vf / radius / wp
    alpha = r0 / radius
    beta = calc_beta(field_intensity, V0, radius)
    #
    epsD = np.linspace(epsDmin, epsDmax, NepsD)

    par_dip = ClusPar(nu_dip, alpha, epsD[0], epsInf)

    # Preparations
    w0 = FrFnd(par_dip).getResonanceFrequencies(1, 1).real
    w_range = w0 + np.linspace(-10, 10, Nw) * nu_dip
    max_losses = np.zeros_like(epsD)
    max_losses_freq = np.zeros_like(epsD)
    mult_losses: List[np.ndarray] = [np.zeros_like(epsD) for i in range(3)]

    losses: List[np.ndarray] = [np.zeros(Nw) for i in range(3)]

    #
    filename = Path(__file__).parent / "na_sphere_savedResults" / hash_str[i]

    datafile = shelve.open(str(filename))
    max_losses = datafile["max_losses"]
    max_losses_freq = datafile["max_losses_freq"]
    mult_losses = datafile["mult_losses"]
    datafile.close()

    ax.plot(epsD, max_losses, label='max loss',color=colors[i])
    for m, losses in enumerate(mult_losses):
        if i == 0:
            ax.plot(epsD, losses, color=colors[i], linestyle=linestyles[m])
        else:
            if m == 1:
                ax.plot(epsD, losses, color=colors[i], linestyle=linestyles[m])

ax.grid(True)
ax.set_xlim((1,1.7))
ax.set_ylim((0,20))
ax.set_xlabel(r"$\varepsilon_d$", fontsize=12)
ax.set_ylabel(r"$Q_{\max}/\omega_pE_0^2$", fontsize=12)
plt.show()
