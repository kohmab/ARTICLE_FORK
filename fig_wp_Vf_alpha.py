import numpy as np
from matplotlib import pyplot as plt

_h = 1.054e-27
_hev = 6.6e-16
_m = 0.91e-27
_e = 4.8e-10
_pi = np.pi


def _Vf(N):
    return _h / _m * (3 * _pi ** 2 * N) ** (1 / 3)


def _wp(N):
    return np.sqrt(4 * _pi * _e ** 2 * N / _m)


def _N_wp(wp):
    return _m * wp ** 2 / 4 / _pi / _e ** 2


def _N_Vf(Vf):
    return (_m / _h) ** 3 * Vf ** 3 / (3 * _pi ** 2)


class DensityRelatedValues:
    _keys = {'wp', 'Vf', 'N'}
    _values = {k: None for k in _keys}
    _calculators = {
        'N': lambda N: {'wp': _wp(N), 'Vf': _Vf(N), 'N': N},
        'wp': lambda wp: {'wp': wp, 'Vf': _Vf(_N_wp(wp)), 'N': _N_wp(wp)},
        'Vf': lambda Vf: {'wp': _wp(_N_Vf(Vf)), 'Vf': Vf, 'N': _N_Vf(Vf)},
    }

    def __init__(self, **kwargs):
        if len(kwargs.keys()) != 1:
            raise TypeError('Incorrect number of arguments')
        key = list(kwargs.keys())[0]
        if key not in self._keys:
            raise KeyError(f'Keyword argument not valid, possible arguments are: {self._keys}')
        self._values[key] = kwargs[key]
        self._recalculate_basing_on(key)

    def _recalculate_basing_on(self, key):
        self._values = self._calculators[key](self._values[key])

    def __getattr__(self, item):
        if item in self._keys:
            return self._values[item]
        raise AttributeError(f'Attribute {item} not found')

    def __setattr__(self, key, value):
        if key in self._keys:
            self._values[key] = value
            self._recalculate_basing_on(key)
        else:
            super().__setattr__(key, value)


if __name__ == '__main__':
    hwev = np.linspace(5, 10, 100)
    a = 5e-7
    drv = DensityRelatedValues(wp=hwev / _hev)
    fig, ax = plt.subplots()
    ax.plot(hwev, drv.N / 1e22, 'b', label='N / 1e22')
    ax.plot(hwev, drv.Vf / 1e8, 'k', label='Vf / 1e8')
    ax.plot(hwev, np.sqrt(3 / 5) * drv.Vf / drv.wp / a * 100, 'r', label='r0/a * 100')
    fig.legend()
    plt.show()

