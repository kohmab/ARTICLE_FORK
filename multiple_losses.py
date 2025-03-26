import pickle
from dataclasses import dataclass
from typing import Tuple, List, Dict

from matplotlib import pyplot as plt

from ClusterParameters import ClusterParameters
from freqFinder import FreqFinder
from losses import generateFilename
import numpy as np

from losses import wp, nu, Vf, epsInf, a, beta, N, wmin, wmax, Nw

# colors = [
#     "#28A745",  # Ярко-зеленый
#     "#007BFF",  # Ярко-синий
#     "#FF4136",  # Ярко-красный
#     "#2B2B2B"  # Угольный
# ]

linestyles = [
    "--", ":", "-."
]

# colors = ['#A22C2E', '#D95A19', '#ECB10A', '#77AB30', '#00B3BD', '#4DBDEC', '#A22C2E']
colors = ['#D95A19', '#ECB10A', '#77AB30', '#00B3BD']


@dataclass(frozen=True)
class FigParams:
    x_lim: Tuple[float, float]
    y_lim: Tuple[float, float]
    epsDk: List[float]
    x_ticks: List[float]
    y_ticks: List[float]


# Nu, alpha, epsInf, beta
known_fig_params: Dict[Tuple, FigParams] = \
    {
        (0.02, 0.038342535127453434, 2, 0.1):
            FigParams(x_lim=(0.32, 0.48),
                      y_lim=(0, 4.5),
                      epsDk=[4, 3, 2.5, 2],  # , 1.4]

                      x_ticks=[0.35, 0.4, 0.45, ],
                      y_ticks=[0, 1, 2, 3, 4],
                      )
    }

if __name__ == '__main__':
    V0 = np.sqrt(3 / 5) * Vf
    r0 = V0 / wp
    alpha = r0 / a

    params_key = (nu, alpha, epsInf, beta)
    fig_params = known_fig_params[params_key] if params_key in known_fig_params else None

    epsDk = fig_params.epsDk if fig_params else [6, 5, 4, 3, 2, 1]

    fig, ax = plt.subplots(figsize=(5, 4))
    for e, epsD in enumerate(epsDk):
        filename = "./savedResults/" + \
                   generateFilename(nu, alpha, epsD, epsInf, beta, N, Nw, wmin, wmax)
        print(nu, alpha, epsD, epsInf, beta, N, Nw, wmin, wmax)
        with open(filename, "rb") as handle:
            w, losses = pickle.load(handle)

        for i, l in enumerate(losses):
            plt.plot(w, l, color=colors[e], linestyle=linestyles[i])
        ax.plot(w, sum(losses), color=colors[e], linestyle='-')
        ax.set_xlim((wmin, wmax))
        ax.grid()
        text = f"bt={beta}_wp={wp * 6.6e-16}ev_de={nu}_epsInf={epsInf}_Vf={Vf / 1e8}(1e8cms)_a={a * 1e7}nm_epsD={epsD}"
        ax.set_title(text)
        filename = f"./loss_res_curves/" + text + ".png"
        fig.savefig(filename)


    if fig_params:
        pos1 = ax.get_position()
        pos2 = [pos1.x0 + 0.025, pos1.y0 + 0.017, pos1.width * 1.027, pos1.height * 1.11]
        # ax_f_1 = fig.add_subplot(111)
        # ax_f_2 = ax_f_1.twinx()
        #
        ax.set_position(pos2)
        # ax_f_1.set_position(fig_params.subAxPos)
        # ax_f_2.set_position(fig_params.subAxPos)
        # NepsD = 30
        # Nres = 10
        # epsD_freqs = np.linspace(*fig_params.EpsD_lim, NepsD)
        # resFreq = np.zeros((NepsD, 3, Nres))

        # for i, ed in enumerate(epsD_freqs):
        #     ff.epsD = ed
        #     for j in range(3):
        #         resFreq[i, j, :] = np.real(ff.getResonanceFrequencies(n=j, Nz=Nres))
        #
        # ax_f_1.plot(epsD_freqs, resFreq[:, 1, 0], colors[1])

        # for m in (0, 2):
        #     for i in range(1, Nres):
        #         freqs = resFreq[:, m, i]
        #         ax_f_2.plot(epsD_freqs, freqs, colors[m])

        ax.set_xlim(fig_params.x_lim)
        ax.set_ylim(fig_params.y_lim)
        ax.set_xticks(fig_params.x_ticks)
        ax.set_yticks(fig_params.y_ticks)
        # ax_f_1.set_ylim(fig_params.w_res_lim)
        # ax_f_2.set_ylim(tuple([p * 2 for p in fig_params.w_res_lim]))

        ax.grid(True)
        # ax_f_1.grid(True)
        # ax_f_2.grid(True)
        ax.set_title("")
    ax.set_xlabel(r"$\omega/\omega_p$", fontsize=12)
    ax.set_ylabel(r"$\dfrac{Q}{\omega_p E_0^2}$", fontsize=12)
    plt.show()
