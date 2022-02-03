import numpy as np
from numpy.polynomial import Polynomial as poly
from scipy import constants as const
from scipy.interpolate import interp1d, interp2d, griddata
from scipy.optimize import minimize_scalar

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
        elif isinstance(r, (int, np.int64)):
            (self.r, self.x, self.y, self.tria, self.den) = (np.zeros((r,)) for _ in range(5))
            self.tria = np.ones((r, 1))
        elif isinstance(r, (tuple, list)):
            if len(r) == 2:
                self.r = np.linspace(0, r[0], r[1])
                (self.x, self.y, self.tria, self.den) = (np.zeros((r[1],)) for _ in range(4))
                self.tria = np.ones((r[1], 1))
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
            theta = np.linspace(0, 2 * np.pi, n_theta + 1)[:-1]

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

    def on_grid(self, *, x=None, y=None, nx=None, ny=None, xdim=None, ydim=None, theta=None):

        den, xx, yy = self.on_self_cartesian(theta=theta)

        if not (x is None):
            x = np.atleast_1d(x)
            if x.ndim == 1:
                nx = len(x)
                xdim = x[-1] - x[0]
            elif x.ndim >= 1:
                nx = x.shape[1]
                xdim = x[0, -1] - x[0, 0]

        if not (y is None):
            y = np.atleast_1d(y)
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

        if x.size != y.size or xdim is None:
            x, y = np.meshgrid(x, y)

        n = griddata((xx.flatten(), yy.flatten()), den.flatten(), (x, y),
                     method='linear')

        return np.squeeze(n), np.squeeze(x), np.squeeze(y)

    def on_polar(self, *, r=None, theta=None):

        if not (r is None) and not (theta is None):
            if isinstance(theta, (int, np.int64)):
                theta = np.linspace(0, 2 * np.pi, theta + 1)[:-1]

            r = np.atleast_1d(r)
            theta = np.atleast_1d(theta)
            if r.size != theta.size:
                r, theta = np.meshgrid(r, theta)

            x = np.squeeze(r * np.cos(theta))
            y = np.squeeze(r * np.sin(theta))
        else:
            x = None
            y = None

        den, x, y = self.on_grid(x=x, y=y, theta=theta)

        # to complex
        vec = x + 1j * y

        return den, np.abs(vec), np.angle(vec), x, y

    def r_lcs(self, limiter):
        n = self.den
        self.den = self.r
        if np.isscalar(limiter):
            _, _, theta = self.on_self_polar()
            theta = theta[:, 0]
            #TODO fix if r.max==rlcs works bad
            rho, r, theta, x, y = self.on_polar(r=limiter, theta=theta)
        # TODO limiter as curve

        ind = np.argmin(rho)

        self.den = n
        return rho[ind], x[ind], y[ind], theta[ind]

    def map_on(self, x, y, den):
        n = self.den
        self.den = self.r

        r, x, y = self.on_grid(x=x, y=y)

        # r must have by construction same length as den
        # remove points outside radius range
        den = np.delete(den, np.isnan(r))
        r = np.delete(r, np.isnan(r))

        # sort by r
        ind = np.argsort(r)
        r = r[ind]
        den = den[ind]
        del ind

        # averaging on r grid to reduce noise and avoid duplicate r values
        nr = len(self.r)
        rr, nn = (np.ndarray((0,)) for _ in range(2))

        for r_low, r_hgh in zip(np.insert((self.r[:-1] + self.r[1:])/2, 0, 0),
                                np.append((self.r[:-1] + self.r[1:])/2, self.r[-1] + (self.r[-1]-self.r[-2])/2)):
            bl = np.logical_and(r_low <= r, r < r_hgh)
            if any(bl):
                rr = np.append(rr, np.mean(r[bl]))
                nn = np.append(nn, np.mean(den[bl]))

        r2den = interp1d(rr, nn, kind='linear', bounds_error=False, fill_value=np.nan)
        self.den = r2den(self.r)
        self.den[np.logical_and(np.isnan(self.den), self.r <= rr.min())] = r2den(rr.min())
        self.den[np.logical_and(np.isnan(self.den), self.r >= rr.max())] = den.min()

    def normalize(self):

        rr = np.zeros_like(self.r)
        for sgn in (-1, 1):
            # make equidistant on high/low field size
            x = sgn * self.r + self.x
            xx = np.linspace(x.min(), x.max(), len(self.r))
            x2r = interp1d(x, self.r, kind='linear', bounds_error=False)
            if sgn > 0:
                rr += x2r(xx)
            else:
                rr += np.flip(x2r(xx))

        rr = rr / 2

        r2x = interp1d(self.r, self.x)
        r2y = interp1d(self.r, self.y)
        r2tria = interp1d(self.r, self.tria)
        r2elon = interp1d(self.r, self.elon)
        r2den = interp1d(self.r, self.den)

        self.r = rr
        self.x = r2x(rr)
        self.y = r2y(rr)
        self.tria = r2tria(rr)
        self.elon = r2elon(rr)
        self.den = r2den(rr)

    def append(self, r, n=5, bkg=1e12):
        if r <= self.r[-1]:
            return

        dn = (self.den[-1] - self.den[-2]) / (self.r[-1] - self.r[-2])
        b = -dn / (self.den[-1] - bkg)
        a = (self.den[-1] - bkg) / np.exp(-b * self.r[-1])
        radd = np.linspace(self.r[-1], r, n + 1)[1:]
        nadd = a * np.exp(-b * radd) + bkg
        self.r = np.concatenate((self.r, radd))
        self.den = np.concatenate((self.den, nadd))
        self.x = np.concatenate((self.x, np.repeat(self.x[-1], n)))
        self.y = np.concatenate((self.y, np.repeat(self.y[-1], n)))
        self.tria = np.concatenate((self.tria, np.repeat(self.tria[-1], n)))
        self.elon = np.concatenate((self.elon, np.repeat(self.elon[-1], n)))


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


def den2phase(den, x=None, y=None, freq=None, axis=0, length_unit='cm', den_unit='cm^-3'):
    # use wave length 2.2 mm
    if freq is None:
        freq = const.speed_of_light * 100 / 0.22

    # coefficient for  length units conversion
    if length_unit == 'cm':
        k = 100
    else:
        k = 1

    den_critical = critical_den(freq, den_unit=den_unit)

    w = 2 * np.pi * freq

    dph_dl = -w / (const.speed_of_light * k) * (np.sqrt(1 - den / den_critical) - 1) / 2 / np.pi

    if x is None and y is None:
        return dph_dl
    if x is None:
        x = np.ones_like(y)
    elif y is None:
        y = np.ones_like(x)

    if np.isscalar(x) and not (np.isscalar(y)):
        x = np.full_like(y, x)
    if np.isscalar(y) and not (np.isscalar(x)):
        y = np.full_like(x, y)

    dl = np.sqrt(np.diff(x, axis=axis) ** 2 + np.diff(y, axis=axis) ** 2)

    # At default axis = 0. It assumes integration along y axis
    if dl.ndim == 2:
        if axis == 1:
            dl = dl.T
            dph_dl = dph_dl.T
        dl = dl[:, 0]
        dph_dl = np.transpose((dph_dl[:-1, :] + dph_dl[1:, :]) / 2)
    else:
        dph_dl = np.transpose((dph_dl[:-1] + dph_dl[1:]) / 2)

    ph = np.dot(dph_dl, dl)

    return ph


def adjust2phase(ph, chord, accuracy=0.01):
    x2ph = interp1d(ph[0], ph[1], kind='cubic', bounds_error=False)

    x = np.atleast_1d(chord[0])[0]
    y = chord[1]
    den = chord[2]

    ph0 = x2ph(x)

    ex = False
    while not ex:
        ph_chord = den2phase(den, x, y)
        den = den * ph0 / ph_chord
        ex = abs(ph0 / ph_chord - 1) < accuracy

    k = np.sum(den) / np.sum(chord[2])
    return den, k


def rxyten_from_params(shifx=0., shify=0., delta=0.5, trian=0., elon=1., num=100, rdia=7.85, length_unit='cm'):

    # convert inputs to polynomial objects
    delta = poly(delta)
    trian = poly(trian)
    elon = poly(elon)

    theta = np.linspace(0, 2 * np.pi, RXYTEN._n_theta_max + 1)[:-1]

    lcs = lambda r: (shifx + delta(r) + r * (np.cos(theta) - trian(r) * np.sin(theta) ** 2)) + \
                    1j * (shify + r * elon(r) * np.sin(theta))

    rlcs = minimize_scalar(lambda r: abs(rdia - np.abs(lcs(r)).max()), bracket=(0, rdia),
                           args=(), method='golden', tol=None,
                           options={'xtol': 1e-4, 'maxiter': 5000}).x

    rxyten = np.zeros((num,6))
    rxyten[:, 0] = np.linspace(0, rlcs, num)
    rxyten[0, 0] = 1e-4
    rxyten[:, 1] = shifx + delta(rxyten[:, 0])
    rxyten[:, 2] = shify
    rxyten[:, 3] = trian(rxyten[:, 0])
    rxyten[:, 4] = elon(rxyten[:, 0])
    # add SOL region
    rxyten = np.vstack((rxyten,np.array([rdia*(1+1/num), 0, 0, 0, 1.0, 0])))
    rxyten = np.vstack((rxyten,np.array([rdia*1.2, 0, 0, 0, 1.0, 0])))

    return RXYTEN(rxyten), rlcs, lcs(rlcs)



