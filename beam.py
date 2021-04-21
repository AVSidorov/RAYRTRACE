import numpy as np
from scipy import constants as const
from scipy.interpolate import interp1d, interp2d
from skimage import measure

from ray import Ray


class Front:
    def __init__(self, x0=0, y0=0, curvature=np.inf, wa=0.0035, len_unit='m'):
        self.x0 = x0
        self.y0 = y0

        self.r = curvature
        self.wa = wa


class Beam:
    _nfront = 1000  # settings for wavefront finder
    _nray = 101  # default number of rays
    _dist = 0.03  # default distance from focus for ray starts
    _ampbnd = 0.001  # default setting for width of rays start area determination
    _gouy = 1  # Gouy phase switch

    def __init__(self, x0=0, y0=0, kx0=0, ky0=1,
                 freq=136.268e9, waist=0.0035, focus=-0.01,
                 n_field=None,
                 freq_unit='Hz', length_unit='m'):
        self.x0 = x0  # antennae x
        self.y0 = y0  # antennae y
        self._kx0 = kx0
        self._ky0 = ky0
        self.freq = freq
        self.waist = waist
        self.focus = focus  # it assumes beam waist doesn't coincide with antennae position
        self._rays = np.ndarray((0,))  # empty array for rays
        self.freq_unit = freq_unit
        self.length_unit = length_unit

        if not (n_field is None):
            self.rays(rnum=Beam._nray, wave_front=None, dist=Beam._dist, ampbnd=Beam._ampbnd, n_field=n_field)

    @property
    def wave_len(self):
        return const.speed_of_light / self.freq  # vacuum wave length

    @wave_len.setter
    def wave_len(self, lw):
        if 0 < lw < 0.1:
            self.freq = const.speed_of_light / lw

    @property
    def kx0(self):
        return self._kx0

    @kx0.setter
    def kx0(self, kx):
        if 1 <= abs(kx):
            self._kx0 = kx / abs(kx)
            self._ky0 = 0
        elif 0 <= abs(kx) < 1:
            self._kx0 = kx
            self._ky0 = self._ky0 / abs(self._ky0) * np.sqrt(1 - kx ** 2)

    @property
    def ky0(self):
        return self._ky0

    @ky0.setter
    def ky0(self, ky):
        if 1 <= abs(ky):
            self._ky0 = ky / abs(ky)
            self._kx0 = 0
        elif 0 <= abs(ky) < 1:
            self._ky0 = ky
            self._kx0 = self._kx0 / abs(self._kx0) * np.sqrt(1 - ky ** 2)

    @property
    def k0(self):
        return self._kx0, self._ky0

    @k0.setter
    def k0(self, k=(0, 1)):
        knorm = np.linalg.norm(k)
        self._kx0 = k[0] / knorm
        self._ky0 = k[1] / knorm

    # converts coordinates from lab to beam frame of reference
    def lab2beam(self, xl, yl):
        # Conversion to beam coordinates

        # shift & rotate
        # wave vector of beam should be normalized
        xb = self._ky0 * (xl - self.x0) - self._kx0 * (yl - self.y0)
        yb = self._kx0 * (xl - self.x0) + self._ky0 * (yl - self.y0)

        # focus is distance along beam propagation axis
        yb = yb - self.focus

        return xb, yb

    # converts coordinates from beam to lab frame of reference
    def beam2lab(self, xb, yb):
        # Conversion to beam coordinates

        # focus
        yb = yb + self.focus

        # rotate
        xl = self._ky0 * xb + self._kx0 * yb
        yl = -self._kx0 * xb + self._ky0 * yb

        # shift
        xl = xl + self.x0
        yl = yl + self.y0

        return xl, yl

    def rayleigh_len(self, n=1.):
        return np.pi * self.waist ** 2 * n / self.wave_len

    # 2D analytical field
    def field(self, *, x=0, y=0, coords=None, n=1., outfmt='field'):
        # if x and y have same sizes (lengths) 1d field will be produced (@ (x,y) points)
        # otherwise result will be produced on grid
        # In case of desired result on a "square" grid, this grid should be given in input
        # otherwise result will be produced on diagonal of desired grid

        # cast input
        x = np.atleast_1d(x)
        y = np.atleast_1d(y)

        # coords have priority
        if coords:
            if isinstance(coords, dict):
                x = np.array(['x'])
                y = np.array(coords['y'])
            elif isinstance(coords, (list, tuple)):
                x = np.array(coords[0])
                y = np.array(coords[1])
            elif isinstance(coords, np.ndarray):
                if coords.shape[1] > coords.shape[0]:
                    coords = coords.T
                if coords.ndim >= 2:
                    x = coords[:, 0]
                    y = coords[:, 1]

        if len(x) != len(y):
            x, y = np.meshgrid(x, y)

        # remove dims if x,y are vectors.
        # If single value for one of coordinate is given, after meshgrid we have matrices with excess axis (size = 1)
        x = np.squeeze(x)
        y = np.squeeze(y)

        # Conversion to beam coordinates
        xx, yy = self.lab2beam(x, y)

        return self._field(xx, yy, n=n, outfmt=outfmt)

    # 2D analytical field in beam coordinates
    def _field(self, xx, yy, n=1., outfmt='field'):
        # constants and parameters
        w0 = self.waist  # waist
        lw = self.wave_len  # vacuum wave length
        y_r = self.rayleigh_len(n=n)  # Rayleigh length

        # y dependent parameters
        wa = w0 * np.sqrt(1. + yy ** 2 / y_r ** 2)  # beam width
        r = np.full_like(yy, np.inf)
        r[yy != 0] = yy[yy != 0] * (1. + y_r ** 2 / yy[yy != 0] ** 2)  # beam curvature radius
        psi = np.arctan(yy / y_r) / 2  # Gouy phase

        # x,y dependent parameters
        amp = np.sqrt(w0 / wa) * np.exp(-xx ** 2 / wa ** 2)
        phase = (2 * np.pi * yy * n / lw + np.pi * xx ** 2 / (r * lw) - psi * Beam._gouy)
        field = amp * np.exp(-1j * phase)

        if outfmt == 'field':
            return field
        elif outfmt == 'tuple':
            return field, amp, phase, r, wa, psi
        elif outfmt == 'dict':
            return {'field': field,
                    'amp': amp,
                    'phase': phase,
                    'r': r,
                    'wa': wa,
                    'gouy': psi
                    }
        else:
            return None

    # wavefront interpolant in beam coordinates
    def _wavefront(self, dist=0.003, ampbnd=0.001, n=1.):
        # constants and parameters
        w0 = self.waist  # waist
        lw = self.wave_len  # wave length
        y_r = self.rayleigh_len(n=n)  # Rayleigh length

        # xmax for grid
        # find x : A(x,dist)/A(0,dist) = ampbnd
        # y fixed = dist
        xmax = np.sqrt(-w0 ** 2 * (1 + (dist / y_r) ** 2) * np.log(ampbnd))
        x = np.linspace(-xmax, xmax, Beam._nfront)

        # find R (wave front curvature) at dist
        # It's necessary for determination grid size in y from xmax
        r = dist * (1 + (y_r / dist) ** 2)

        # loop to avoid too short front. Front must have ends on left/right wall
        nl = 1
        ex = False
        while not ex:
            y = np.linspace(dist - np.sqrt(r ** 2 - xmax ** 2) - nl * lw, dist + nl * lw, Beam._nfront)
            ind2x = interp1d(np.arange(len(x)), x)
            ind2y = interp1d(np.arange(len(y)), y)

            # make 2d grid for front line determination
            xx, yy = np.meshgrid(x, y)

            field_set = self._field(xx, yy, n=n, outfmt='dict')

            ph0 = 2 * np.pi * dist * n / lw - np.arctan(
                dist / y_r) / 2 * Beam._gouy  # phase on beam axis on dist from focus
            contour = measure.find_contours(field_set['phase'], ph0)  # wavefront line in grid coordinates
            if ind2x(contour[0][0, 1]) <= -xmax:
                ex = True
            else:
                nl += 1

        # interpolant from x to y for wavefront in beam coordinates
        wave_front = interp1d(ind2x(contour[0][:, 1]), ind2y(contour[0][:, 0]), kind='cubic')
        return wave_front

    # wavefront in lab coordinates
    def wavefront(self, dist=0.02, ampbnd=0.001, n=1.):
        wave_front = self._wavefront(dist, ampbnd, n=n)
        return self.beam2lab(wave_front.x, wave_front.y)

    # wave front for rays generator
    def wavefront_r(self, dist=0.03, ampbnd=0.001, n_field=None):
        # constants and parameters
        if n_field is None:
            n = 1.  # index of refraction
        else:
            n = n_field.refractive_index(freq=self.freq)
        lw = self.wave_len  # wave length
        y_r = self.rayleigh_len(n=n)  # Rayleigh length

        # determinate nearest to dist y there phase = n * 2 * pi
        # number of periods
        # nl differs from number of wave lengths in dist due to Gouy phase
        nl = np.floor(n * dist / lw - Beam._gouy * np.arctan(dist / y_r) / 2 / 2 / np.pi)

        # new corrected dist determination
        y = np.linspace((nl - 1) * lw, (nl + 1) * lw, 100)
        n2y = interp1d(n * y / lw - Beam._gouy * np.arctan(y / y_r) / 2 / 2 / np.pi, y)
        dist = n2y(nl)

        # getting interpolant of wavefront x to y
        return self._wavefront(dist, ampbnd, n=n)

    def rays(self, rnum=101, wave_front=None, dist=0.03, ampbnd=0.001, n_field=None):

        if wave_front is None:
            wave_front = self.wavefront_r(dist=dist, ampbnd=ampbnd, n_field=n_field)

        # make equidistant ray start points
        # length axis
        lax = np.sqrt(np.diff(wave_front.x) ** 2 + np.diff(wave_front.y) ** 2)
        lax = np.cumsum(np.insert(lax, 0, 0))

        # interpolant from lax to x
        l2x = interp1d(lax, wave_front.x)

        # start points
        x = l2x(np.linspace(0, lax[-1], rnum))
        y = wave_front(x)

        # finding normals

        # make fine equidistant grid (in distance meaning, not x or y)
        xg = l2x(np.linspace(0, lax[-1], np.lcm(rnum, Beam._nfront)))
        yg = wave_front(xg)

        # go to lab coordinates
        # here to avoid wave vector rotations

        x, y = self.beam2lab(x, y)
        xg, yg = self.beam2lab(xg, yg)

        # wave vector components
        kx = -np.gradient(yg)
        ky = np.gradient(xg)

        # normalization of wave vector
        nrmd = np.sqrt(kx ** 2 + ky ** 2)
        kx = kx / nrmd
        ky = ky / nrmd

        # slicing to get rnum rays
        kx = kx[::int(len(kx) / rnum)]
        ky = ky[::int(len(ky) / rnum)]

        field_set = self.field(x=x, y=y, n=n, outfmt='dict')

        amps = field_set['amp']
        phase = field_set['phase']  # for check

        if n_field:
            # coefficient for units conversion
            if self.length_unit == 'm':
                k = 100  # raytracing works in CGS
            else:
                k = 1

            for x0, y0, kx0, ky0, amp0 in zip(x * k, y * k, kx, ky, amps):
                # TODO check ray start inside density grid
                self._rays = np.append(self._rays, Ray(x0, y0, kx0, ky0, self.freq, amp0, n_field=n_field))

        return x, y, kx, ky, amps, phase, dist

    # 1D field obtained from rays
    def field_r(self, y, axis='y'):

        if self.length_unit == 'm':
            y = y * 100  # rays are in CGS
        # TODO parsing for x,y

        amp0, ph, dist0, dist1, xout = (np.ndarray((0,)) for _ in range(5))
        pnt = np.ndarray((0,), dtype=int)

        for rm, rl, rr in zip(self._rays, np.roll(self._rays, 1), np.roll(self._rays, -1)):

            amp0 = np.append(amp0, rm.amp)

            dist0 = np.append(dist0, np.sqrt((rm.x[0] - rl.x[0]) ** 2 + (rm.y[0] - rl.y[0]) ** 2))
            dist0 = np.append(dist0, np.sqrt((rm.x[0] - rr.x[0]) ** 2 + (rm.y[0] - rr.y[0]) ** 2))

            ind = rm.addpoint(y, axis=axis)
            if len(ind) > 1:
                return None

            pnt = np.append(pnt, ind)
            ph = np.append(ph, rm.ph[ind])
            xout = np.append(xout, rm.x[ind])

            xx = np.ndarray((0,))
            yy = np.ndarray((0,))
            for r in (rl, rr):
                ph2x = interp1d(r.ph, r.x, fill_value='extrapolate')
                ph2y = interp1d(r.ph, r.y, fill_value="extrapolate")
                xx = np.append(xx, ph2x(rm.ph[ind]))
                yy = np.append(yy, ph2y(rm.ph[ind]))

            dist1 = np.append(dist1, np.sqrt((xx - rm.x[ind]) ** 2 + (yy - rm.y[ind]) ** 2))

        dist0[0] = dist0[1]
        dist0[-1] = dist0[-2]

        dist1[0] = dist1[1]
        dist1[-1] = dist1[-2]

        # For straight rays distance between points increases as 1/R
        # but in 2D case field amplitude decreases with distance as 1/sqrt(R)
        amp1 = np.sqrt(np.sum(dist0.reshape((-1, 2)), 1) / np.sum(dist1.reshape((-1, 2)), 1))

        field = amp0 * amp1 * np.exp(-1j * ph)

        return field, xout, amp0, amp1, ph, pnt, dist0, dist1


def beam_from_field(field, x, y, dist=0.03, ampbnd=0.01):
    # normalize field
    field = field / np.max(np.abs(field).flatten())

    # working in grid coordinates (dimensionless)
    # grid coordinates
    xi = range(len(x))
    yi = range(len(y))

    # amplitude interpolant
    fieldAbs = interp2d(xi, yi, np.abs(field))

    yi_max, xi_max = np.unravel_index(np.argmax(np.abs(field)), field.shape)

    # find fronts
    fronts = measure.find_contours(np.angle(field), 0)
    full_fronts = np.ndarray((0, 3))

    for ind in range(len(fronts)):
        amp = np.abs(field[np.round(fronts[ind][:, 0]).astype(np.int64),
                           np.round(fronts[ind][:, 1]).astype(np.int64)]).flatten()
        fronts[ind] = np.delete(fronts[ind], amp / amp.max() < ampbnd, 0)
        amp = np.delete(amp, amp / amp.max() < ampbnd, 0)
        full_fronts = np.vstack((full_fronts, np.hstack((fronts[ind], np.atleast_2d(amp).T))))

    full_fronts = full_fronts[:, [1, 0, 2]]
    yimax, ximax = np.where(
        np.logical_and(np.logical_and(field.real >= np.roll(field.real, 1, 0),
                                      field.real >= np.roll(field.real, -1, 0)),
                       np.logical_and(field.real >= np.roll(field.real, 1, 1),
                                      field.real >= np.roll(field.real, -1, 1))))
    # amplitudes at maxima
    ampmax = np.abs(field[yimax, ximax]).flatten()
    index = np.argsort(1 / ampmax)
    ampmax = ampmax[index]
    ximax = ximax[index]
    yimax = yimax[index]

    yimin, ximin = np.where(
        np.logical_and(np.logical_and(field.real <= np.roll(field.real, 1, 0),
                                      field.real <= np.roll(field.real, -1, 0)),
                       np.logical_and(field.real <= np.roll(field.real, 1, 1),
                                      field.real <= np.roll(field.real, -1, 1))))


def points2circle(a, b, c):
    a2 = np.dot(a, a)
    b2 = np.dot(b, b)
    c2 = np.dot(c, c)
    d = np.cross(a - b, a - c)
    if d == 0:
        return np.nan
    e = a * (b2 - c2) + b * (c2 - a2) + c * (a2 - b2)
    r = np.array((-e[1] / d / 2, e[0] / d / 2))
    return r
