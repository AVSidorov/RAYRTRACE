import numpy as np
from scipy import constants as const
from scipy.interpolate import interp1d, interp2d


class Ray:
    def __init__(self, x0=0, y0=0, kx0=0, ky0=1, freq=135e9):
        self.x = np.zeros(10000)
        self.y = np.zeros(10000)
        self.l = np.zeros(10000)
        self.kx = np.zeros(10000)
        self.ky = np.zeros(10000)
        self.ph = np.zeros(10000)
        self.ax = np.zeros(10000)
        self.ay = np.zeros(10000)

        self.x[0] = x0
        self.y[0] = y0

        self.freq = freq
        c = const.speed_of_light * 100
        w = 2 * np.pi * freq
        # if freq is not set try to get frequency from wave number
        if np.sqrt(kx0 ** 2 + ky0 ** 2) == 1:
            # assume that only initial direction of ray is given
            kx0 = kx0 * w / c
            ky0 = ky0 * w / c
        elif self.freq == 135e9:
            # if default value of frequency is used
            w = np.sqrt(kx0 ** 2 + ky0 ** 2) * c
            self.freq = w / 2 / np.pi

        self.kx[0] = kx0
        self.ky[0] = ky0

    def launch(self, n_field):
        # It assumes that grid is equidistant
        import timeit
        start = timeit.default_timer()

        # constants
        w = 2 * np.pi * self.freq
        # working in CGS so additional coefficients should be used
        c = const.speed_of_light * 100
        e = 4.8032e-10
        nc = const.electron_mass * 1000 * w ** 2 / (4 * np.pi * e ** 2)

        # grid size
        nx = len(n_field['x'])
        ny = len(n_field['y'])

        # grid step
        dX = n_field['x'][1] - n_field['x'][0]
        dY = n_field['y'][1] - n_field['y'][0]

        # gradient arrays
        dNx = (n_field['n'][0:-1, 1:] - n_field['n'][0:-1, 0:-1]) / dX
        dNy = (n_field['n'][1:, 0:-1] - n_field['n'][0:-1, 0:-1]) / dY

        # step determination
        dXYmin = min(dX, dY) / const.golden_ratio
        Amax = max(np.max(abs(dNx)), np.max(abs(dNy))) / 2 / nc * w ** 2 / c ** 2
        Bmax = w / c
        t = min((-Bmax + np.sqrt(Bmax ** 2 + 4 * Amax * dXYmin)) / 2 / Amax, dXYmin * c / w)

        # intialization of ray
        # convert to grid coordinates
        Xinterpolant = interp1d(n_field['x'], np.arange(0, nx))
        Yinterpolant = interp1d(n_field['y'], np.arange(0, ny))
        Ninterpolant = interp2d(n_field['x'], n_field['y'], n_field['n'], kind='cubic')
        refrIdx = np.sqrt(1 - Ninterpolant(self.x[0], self.y[0]) / nc)
        self.kx[0] = self.kx[0] * refrIdx
        self.ky[0] = self.ky[0] * refrIdx
        curX = Xinterpolant(self.x[0])
        curY = Yinterpolant(self.y[0])

        i = 0
        while 0 <= curX and nx - 1 > curX and 0 <= curY and ny - 1 > curY:
            self.ax[i] = -dNx[int(np.fix(curY)), int(np.fix(curX))] / 2 / nc * w ** 2 / c ** 2
            self.ay[i] = -dNy[int(np.fix(curY)), int(np.fix(curX))] / 2 / nc * w ** 2 / c ** 2

            self.kx[i + 1] = self.kx[i] + self.ax[i] * t
            self.ky[i + 1] = self.ky[i] + self.ay[i] * t

            self.x[i + 1] = self.x[i] + self.kx[i] * t + self.ax[i] * t ** 2 / 2
            self.y[i + 1] = self.y[i] + self.ky[i] * t + self.ay[i] * t ** 2 / 2

            k = np.sqrt(self.kx[i] ** 2 + self.ky[i] ** 2)
            k1 = np.sqrt(self.kx[i + 1] ** 2 + self.ky[i + 1] ** 2)

            l = t * (k1 + k) / 2
            self.ph[i + 1] = self.ph[i] + l * (k1 + k) / 2
            self.l[i + 1] = self.l[i] + l

            curX = curX + (self.kx[i] * t + self.ax[i] * t ** 2 / 2) / dX
            curY = curY + (self.ky[i] * t + self.ay[i] * t ** 2 / 2) / dY
            i += 1

        self.x = self.x[:i]
        self.y = self.y[:i]
        self.kx = self.kx[:i]
        self.ky = self.ky[:i]
        self.ax = self.ax[:i]
        self.ay = self.ay[:i]
        self.l = self.l[:i]
        self.ph = self.ph[:i]

        stop = timeit.default_timer()
        self.launch_t = stop - start
