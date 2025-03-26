import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn


class LorentzCurve:
    f: float
    w0: float
    gamma: float

    def __init__(self, f: float, w0: float, gamma: float):
        self.f = f
        self.w0 = w0
        self.gamma = gamma

    def __call__(self, w: float) -> complex:
        return self.f / ((w ** 2 - self.w0 ** 2) + 1j * w * self.gamma)


class AlEpsilon:  # https://doi.org/10.1364/AO.34.004755

    def __init__(self):
        self._osc = []

        wp = 14.94  # eV
        self._osc.append(LorentzCurve(wp ** 2 * 0.632, 0, 0.075))
        self._osc.append(LorentzCurve(wp ** 2 * 0.109, 0.34, 0.44))
        self._osc.append(LorentzCurve(wp ** 2 * 0.096, 1.57, 0.45))
        self._osc.append(LorentzCurve(wp ** 2 * 0.122, 2.11, 1.41))
        self._osc.append(LorentzCurve(wp ** 2 * 0.024, 4.59, 2.82))

    def __call__(self, w: float) -> complex:
        result = np.ones_like(w, dtype=complex)
        for o in self._osc:
            result -= o(w)
        return result


class ModelEps(nn.Module):
    def __init__(self, eps_inf_opt=False):
        super().__init__()
        self.wp = nn.Parameter(14.92 * torch.ones(1, dtype=torch.float), requires_grad=True)
        self.nu = nn.Parameter(0.4889 * torch.ones(1, dtype=torch.float), requires_grad=True)
        self.epInf = nn.Parameter(torch.ones(1, dtype=torch.float), requires_grad=True)
        self.epInf.requires_grad_(eps_inf_opt)

    def forward(self, w):
        eps = torch.ones_like(w, dtype=torch.complex64) * self.epInf
        eps -= self.wp ** 2 / w / (w + 1j * self.nu)
        return eps


if __name__ == '__main__':
    aleps = AlEpsilon()
    model = ModelEps()

    w = np.linspace(8, 30, 100)

    eps = aleps(w)

    optim = torch.optim.SGD(model.parameters(), lr=1e-3)
    target_eps = torch.from_numpy(eps)
    loss_fn = torch.nn.MSELoss(reduction='sum')

    fig, ax = plt.subplots()

    er = torch.tensor([100], dtype=torch.float, requires_grad=True)
    step = 0
    while er > 0.01:
        eps_m = model(torch.from_numpy(w))
        er = (eps_m - target_eps).abs().sum()
        er.backward()
        optim.step()
        optim.zero_grad()

        wp = model.wp.detach().numpy()
        nu = model.nu.detach().numpy()
        epsInf = model.epInf.detach().numpy()

        print(er)
        print(f"wp = {wp} eV , nu = {nu} eV, epsInf = {epsInf}")

        step += 1
        if step % 15 == 0:
            eps_m_plot = eps_m.detach().numpy()
            ax.cla()
            ax.plot(w, eps.real, 'r', label='real')
            ax.plot(w, eps.imag, 'b', label='imag')
            ax.plot(w, eps_m_plot.real, 'r--', label='real model')
            ax.plot(w, eps_m_plot.imag, 'b--', label='imag model')
            ax.grid(True)
            plt.pause(0.00001)
            plt.draw()

plt.show()
