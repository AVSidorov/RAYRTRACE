import numpy as np
from scipy import constants as const
from scipy.interpolate import interp1d, interp2d, griddata

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


class RXYTEN:
    _n_theta_max = 1000

    def __init__(self, r, x=None, y=None, tria=None, elon=None, den=None):
        if isinstance(r, np.ndarray):
            if r.ndim == 2:
                if r.shape[1] == 6:
                    r, x, y, tria, elon, den = (r[:, i] for i in range(6))
        ind = np.argsort(r)
        self.r = r[ind]
        self.x = x[ind]
        self.y = y[ind]
        self.tria = tria[ind]
        self.elon = elon[ind]
        self.den = den[ind]

    def on_self_polar(self, theta=None):

        if theta is None:
            grid_step = np.diff(self.r).min()
            theta = np.round(2 * np.pi * self.r.max() / grid_step).astype(np.int64)
            theta = min(RXYTEN._n_theta_max, theta)

        if isinstance(theta, (int, np.int64)):
            n_theta = theta
            theta = np.linspace(0, 2*np.pi, n_theta+1)[:-1]

        den, _ = np.meshgrid(self.den, theta)
        r, theta = np.meshgrid(self.r, theta)

        return den, r, theta

    def on_self_cartesian(self, theta=None):
        den, r, theta = self.on_self_polar(theta=theta)
        xx = self.x + r * (np.cos(theta) - self.tria * np.sin(theta)**2)
        yy = self.y + r * self.elon * np.sin(theta)

        return den, xx, yy

    @property
    def nested(self):
        _, xx, yy = self.on_self_cartesian()
        inout = 1
        for ri in range(len(self.r) - 1):
            inout *= -check_inside((xx[:, ri] - self.x[ri], yy[:, ri] - self.y[ri]),
                                   (xx[:, ri + 1] - self.x[ri], yy[:, ri + 1] - self.y[ri]))
        if inout == 1:
            return True
        else:
            return False

    def on_grid(self, *, x=None, y=None, nx=None, ny=None, xdim=None, ydim=None):

        den, xx, yy = self.on_self_cartesian()

        x = np.atleast_1d(x)
        y = np.atleast_1d(y)

        if not(x is None):
            if x.ndim == 1:
                nx = len(x)
                xdim = x[-1]-x[0]
            elif x.ndim >= 1:
                nx = x.shape[1]
                xdim = x[0, -1] - x[0, 0]

        if not(y is None):
            if y.ndim == 1:
                ny = len(y)
                ydim = y[-1] - y[0]
            elif y.ndim >= 1:
                ny = y.shape[0]
                ydim = y[-1, 0] - y[0, 0]

        if nx is None:
            nx = len(self.r)

        if ny is None:
            ny = nx

        if xdim is None:
            x = np.linspace(xx.min(), xx.max(), nx)
        elif x is None:
            x = np.linspace(-xdim / 2, xdim / 2, nx)

        if ydim is None:
            y = np.linspace(yy.min(), yy.max(), ny)
        elif y is None:
            y = np.linspace(-ydim / 2, ydim / 2, nx)

        if x.ndim == 1:
            x, y = np.meshgrid(x, y)

        n = griddata((xx.flatten(), yy.flatten()), den.flatten(), (x, y),
                     method='cubic', fill_value=self.den.min())

        return n, x, y


def check_inside(a, b):
    # for this algorithm origin must be inside of convex contour
    # for working in polar coordinates make complex numbers
    a = a[0] + 1j * a[1]
    b = b[0] + 1j * b[1]

    a = a[np.argsort(np.angle(a))]
    b = b[np.argsort(np.angle(b))]

    ang2rho = interp1d(np.angle(b), np.abs(b), kind='cubic', bounds_error=False,
                       fill_value=(np.abs(b[0]), np.abs(b[-1])))

    # radius coordinates of "b" on same angels as in "a"
    rb = ang2rho(np.angle(a))

    if all(np.abs(a) < rb):
        return -1  # a is inside b
    elif all(np.abs(a) > rb):
        return 1  # a is outside b
    else:
        return 0
