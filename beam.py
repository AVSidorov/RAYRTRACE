import numpy as np
from scipy import constants as const
from scipy.interpolate import interp1d
from skimage import measure

from ray import Ray


class Beam:
    _nfront = 1000  # settings for wavefront finder
    _nray = 100
    _dist = 0.02
    _ampbnd = 0.001

    def __init__(self, x0=0, y0=0, kx0=0, ky0=1,
                 freq=136.268e9, waist=0.0035, focus=0.01,
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

        if n_field:
            self.rays(rnum=Beam._nray, dist=Beam._dist, ampbnd=Beam._ampbnd, n_field=n_field)

    @property
    def wave_len(self):
        return const.speed_of_light / self.freq  # wave length

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
    def k(self):
        return self._kx0, self._ky0

    @k.setter
    def k(self, k=(0, 1)):
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

    # 2D field in vacuum
    def field(self, *, x=np.array(0), y=np.array(0), coords=None, outfmt='field'):
        # if x and y have same sizes (lengths) 1d field will be produced (@ (x,y) points)
        # otherwise result will be produced on grid
        # In case of desired result on a "square" grid, this grid should be given in input
        # otherwise result will be produced on diagonal of desired grid

        # cast input in case if not default values are given
        x = np.array(x)
        y = np.array(y)

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

        # Conversion to beam coordinates
        xx, yy = self.lab2beam(x, y)

        return self._field(xx, yy, outfmt=outfmt)

    # 2D vacuum field in beam coordinates
    def _field(self, xx, yy, outfmt='field'):
        # constants and parameters
        w0 = self.waist  # waist
        lw = const.speed_of_light / self.freq  # wave length
        y_r = np.pi * w0 ** 2 / lw  # Rayleigh length

        # y dependent parameters
        wa = w0 * np.sqrt(1. + yy ** 2 / y_r ** 2)  # beam width
        r = np.full_like(yy, np.inf)
        r[yy != 0] = yy[yy != 0] * (1. + y_r ** 2 / yy[yy != 0] ** 2)  # beam curvature radius
        psi = np.arctan(yy / y_r)  # Gouy phase

        # x,y dependent parameters
        amp = w0 / wa * np.exp(-xx ** 2 / wa ** 2)
        phase = (2 * np.pi * yy / lw + np.pi * xx ** 2 / (r * lw) - psi)
        field = amp * np.exp(1j * phase)

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
    def _wavefront(self, dist=0.02, ampbnd=0.001):
        # constants and parameters
        w0 = self.waist  # waist
        lw = self.wave_len  # wave length
        y_r = np.pi * w0 ** 2 / lw  # Rayleigh length

        # xmax for grid
        # find x : A(x,dist)/A(0,dist) = ampbnd
        # y fixed = dist
        xmax = np.sqrt(-w0 ** 2 * (1 + dist ** 2 / y_r ** 2) * np.log(ampbnd))

        # make 2d grid for front line determination
        x = np.linspace(-xmax, xmax, Beam._nfront)
        # TODO if ampbnd is very small y low border should be move toward beam focus
        y = np.linspace(dist - lw, dist + lw, Beam._nfront)
        xx, yy = np.meshgrid(x, y)

        field_set = self._field(xx, yy, outfmt='dict')

        ph0 = 2 * np.pi * dist / lw - np.arctan(dist / y_r)  # phase on beam axis on dist from focus
        contour = measure.find_contours(field_set['phase'], ph0)  # wavefront line in grid coordinates
        ind2x = interp1d(np.arange(len(x)), x)
        ind2y = interp1d(np.arange(len(y)), y)

        # interpolant from x to y for wavefront in beam coordinates
        wave_front = interp1d(ind2x(contour[0][:, 1]), ind2y(contour[0][:, 0]))
        return wave_front

    # wavefront in lab coordinates
    def wavefront(self, dist=0.02, ampbnd=0.001):
        wave_front = self._wavefront(dist, ampbnd)
        return self.beam2lab(wave_front.x, wave_front.y)

    def rays(self, rnum=100, dist=0.02, ampbnd=0.001, n_field=None):

        # constants and parameters
        w0 = self.waist  # waist
        lw = self.wave_len  # wave length
        y_r = np.pi * w0 ** 2 / lw  # Rayleigh length

        # determinate nearest to dist y there phase = n * 2 * pi
        # number of periods
        # nl differs from number of wave lengths in dist due to Gouy phase
        nl = np.floor(dist / lw - np.arctan(dist / y_r) / 2 / np.pi)

        # new corrected dist determination
        y = np.linspace((nl - 1) * lw, (nl + 1) * lw, 100)
        n2y = interp1d(y / lw - np.arctan(y / y_r) / 2 / np.pi, y)
        dist = n2y(nl)

        # getting interpolant of wavefront x to y
        wave_front = self._wavefront(dist, ampbnd)

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

        field_set = self.field(x=x, y=y, outfmt='dict')

        amps = field_set['amp']
        phase = field_set['phase']  # for check

        if n_field:
            # coefficient for units conversion
            if self.length_unit == 'm':
                k = 100
            else:
                k = 1

            for x0, y0, kx0, ky0, amp0 in zip(x * k, y * k, kx, ky, amps):
                # TODO check ray start inside density grid
                self._rays = np.append(self._rays, Ray(x0, y0, kx0, ky0, self.freq, amp0, n_field=n_field))

        return x, y, kx, ky, amps, phase, dist
