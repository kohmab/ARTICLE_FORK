from ClusterParameters import ClusterParameters
from freqFinder import FreqFinder
import numpy as np
import matplotlib.pyplot as plt

import matplotlib.patches as patches
from matplotlib.path import Path

from tqdm import tqdm

# XXX
# wp = 5 / 6.6e-16
# nu = 0.02
# Vf = 1.4e8
# epsInf = 3
# a = 3.5e-7

# XXX
from losses import wp, nu, Vf, epsInf, a

V0 = np.sqrt(3 / 5) * Vf
r0 = V0 / wp
alpha = r0/a


filename = f"./freq_dep/al={alpha}_nu={nu}_eps={epsInf}.png"
par = ClusterParameters(nu, alpha, 1, epsInf)

ff = FreqFinder(parameters=par)


def patch_from_dep(xs, ys, color):
    verts = []
    for pair in zip(xs - nu / 2, ys):
        verts.append(pair)
    for pair in zip(xs[::-1] + nu / 2, ys[::-1]):
        verts.append(pair)
    verts.append(verts[0])
    codes = [Path.LINETO] * len(verts)
    codes[0] = Path.MOVETO
    codes[-1] = Path.CLOSEPOLY
    path = Path(verts, codes)
    return patches.PathPatch(path, facecolor=color, lw=0.01, alpha=0.33)


Nres = 10
NepsD = 100
epsD = np.linspace(1, 3. * epsInf, NepsD)
limw = np.array([1/np.sqrt(epsInf+epsD[-1]) - 0.1, 1/np.sqrt(epsInf+epsD[0]) + 0.1])

resFreq = np.zeros((NepsD, 3, Nres))



for i, ed in tqdm(enumerate(epsD), total=NepsD):
    ff.epsD = ed
    for j in range(3):
        resFreq[i, j, :] = np.real(ff.getResonanceFrequencies(n=j, Nz=Nres))

fig, host = plt.subplots(figsize=(8, 5), layout='constrained')

ax1 = host.twiny()
colors = {0: 'green', 1: 'blue', 2: 'red'}

host.plot(resFreq[:, 1, 0], epsD, colors[1])

for m in (0, 2):
    for i in range(0, Nres):
        freqs = resFreq[:, m, i]
        ax1.plot(freqs, epsD, colors[m])
        patch = patch_from_dep(freqs, epsD, colors[m])
        ax1.add_patch(patch)

ax1.grid(True)

host.set_title(fr"$\nu/\omega_p = {nu} , r_0 / a = {alpha}, \varepsilon_\infty = {epsInf}$")
host.grid(True)
host.set_xlim(limw)
ax1.set_xlim(2 * limw)
fig.savefig(filename)
plt.show()
