import numpy as np
from scipy import constants as const
from scipy.interpolate import interp1d, interp2d

from beam import Beam


class Beams(list):
    def __init__(self, n_field=None):
        self.n_field = n_field
        super().__init__()

    def append(self, *args, **kwargs) -> None:
        super().append(Beam(*args, **kwargs, n_field=self.n_field))
        # TODO freq from nfield


def critical_den(freq=136.29e9, den_unit='cm^-3'):
    # constants
    # working in CGS so additional multipliers should be used
    w = 2 * np.pi * freq
    c = const.speed_of_light * 100
    e = 4.8032e-10

    k = 1
    if den_unit == 'cm^-3':
        k = 1
    elif den_unit == 'm^-3':
        k = 1e6

    return k * const.electron_mass * 1000 * w ** 2 / (4 * np.pi * e ** 2)


class Nfield:

    def __init__(self, den, *, nx=100, ny=100, x=None, y=None,
                 freq=136.26e9, len_unit='cm', den_unit='cm^-3'):

        self.len_unit = len_unit
        self.den_unit = den_unit

        # TODO: add support of rxyten like density input
        # TODO: error checking
        if isinstance(den, dict):
            if 'n' in den.keys():
                self.n = np.array(den['n'], np.float64)
            else:
                pass
            if 'x' in den.keys():
                self.x = np.array(den['x'])
                self.nx = len(self.x)
            if 'y' in den.keys():
                self.y = np.array(den['y'])
                self.ny = len(self.y)
        elif isinstance(den, np.ndarray):
            if den.ndim == 2:
                self.n = np.array(den, np.float64)
                self.ny = den.shape[0]  # should be row number
                self.nx = den.shape[1]
                if x is None:
                    self.x = np.arange(self.nx)
                else:
                    # cast to ndarray
                    x = np.atleast_1d(x)
                    if len(x) != self.nx and x.size != den.size:
                        # TODO raise exception
                        return
                    else:
                        self.x = x

                if y is None:
                    self.y = np.arange(self.ny)
                else:
                    # cast to ndarray
                    y = np.atleast_1d(y)
                    if len(y) != self.ny and y.size != den.size:
                        return
                    else:
                        self.y = y

        # TODO: universal field inner state must be hidden. (2D grid should be stored inside 2D interpolant)
        # TODO methods that return field in desired way (coordinate system, coordinates and so on)
        # TODO: units conversion

        # grid step
        self.dx = self.x[1:] - self.x[:-1]
        self.dy = self.y[1:] - self.y[:-1]

        dxx, dyy = np.meshgrid(self.dx, self.dy)

        # gradients
        self.dn_x = (self.n[:-1, 1:] - self.n[:-1, :-1]) / dxx
        self.dn_y = (self.n[1:, :-1] - self.n[:-1, :-1]) / dyy

        # for conversion to grid coordinates
        self.x2grid = interp1d(self.x, np.arange(0, self.nx), fill_value='extrapolate')
        self.y2grid = interp1d(self.y, np.arange(0, self.ny), fill_value='extrapolate')
        self.n_interpolant = interp2d(self.x, self.y, self.n, kind='cubic', fill_value=1e12)

        self.min_dxy = min(self.dx.min(), self.dy.min())
        self.prep_raytrace(freq=freq)
        self.beamsTX = Beams(self)
        self.beamsRX = Beams()

    def refractive_index(self, freq=136.26e9, x=None, y=None):
        if x is None or y is None:
            den = self.n[0, 0]  # take density at boundary
        else:
            den = self.n_interpolant(x, y)
        nc = critical_den(freq, den_unit=self.den_unit)
        return np.sqrt(1 - den / nc)

    def prep_raytrace(self, freq=136.26e9):
        # TODO separate class for arrays necessary for  raytracing
        self.freq = freq

        # constants
        # working in CGS so additional multipliers should be used
        w = 2 * np.pi * self.freq
        c = const.speed_of_light * 100
        nc = critical_den(freq, den_unit=self.den_unit)

        self.ax = -self.dn_x / 2 / nc * (w / c) ** 2
        self.ay = -self.dn_y / 2 / nc * (w / c) ** 2

        # TODO variable step (for each cell)
        # step determination
        a_max = max(np.max(abs(self.ax)), np.max(abs(self.ay)))
        b_max = w / c

        if a_max == 0:
            st_grad = np.inf
        else:
            st_grad = (-b_max + np.sqrt(b_max ** 2 + 4 * a_max * self.min_dxy)) / 2 / a_max

        self.st = min(st_grad, self.min_dxy * c / w)
