import numpy as np
from scipy import constants as const


class Ray:
    MaxStep = 100000  # Settings variable for arrays preallocating

    def __init__(self, x0=0, y0=0, kx0=0, ky0=1, freq=135e9, amp=1, n_field=None):
        self.x = np.zeros(Ray.MaxStep)
        self.y = np.zeros(Ray.MaxStep)
        self.kx = np.zeros(Ray.MaxStep)
        self.ky = np.zeros(Ray.MaxStep)

        self.x[0] = x0
        self.y[0] = y0

        self.freq = freq
        c = const.speed_of_light * 100
        w = 2 * np.pi * freq

        if np.sqrt(kx0 ** 2 + ky0 ** 2) <= 2 or self.freq != 135e9:
            # only initial direction of ray is given
            # or freq is fixed

            # normalize wave vector
            k = np.linalg.norm((kx0, ky0))
            kx0 = kx0 / k
            ky0 = ky0 / k

            # make vacuum value
            kx0 = kx0 * w / c
            ky0 = ky0 * w / c

        # if freq is not set try to get frequency from wave number
        elif self.freq == 135e9:
            # if default value of frequency is used
            w = np.sqrt(kx0 ** 2 + ky0 ** 2) * c
            self.freq = w / 2 / np.pi

        self.kx[0] = kx0
        self.ky[0] = ky0

        self.amp = amp

        # launch ray if density is given
        if not (n_field is None):
            # change wave vector corresponding to local density
            refr = np.sqrt(1 - n_field.n_interpolant(self.x[0], self.y[0]) / n_field.nc)
            self.kx[0] = self.kx[0] * refr
            self.ky[0] = self.ky[0] * refr

            # It assumes that grid is equidistant
            # TODO: checking of equidistant if not then recalc nfield
            dx = n_field.dx[0]
            dy = n_field.dy[0]

            # convert to grid coordinates
            x_cur = n_field.x2grid(self.x[0])
            y_cur = n_field.y2grid(self.y[0])

            i = 0
            st = n_field.st
            while 0 <= x_cur < n_field.nx - 1 and 0 <= y_cur < n_field.ny - 1:
                x_ind = int(np.fix(x_cur))
                y_ind = int(np.fix(y_cur))

                self.kx[i + 1] = self.kx[i] + n_field.ax[y_ind, x_ind] * st
                self.ky[i + 1] = self.ky[i] + n_field.ay[y_ind, x_ind] * st

                self.x[i + 1] = self.x[i] + self.kx[i] * st + n_field.ax[y_ind, x_ind] * st ** 2 / 2
                self.y[i + 1] = self.y[i] + self.ky[i] * st + n_field.ay[y_ind, x_ind] * st ** 2 / 2

                x_cur = x_cur + (self.x[i + 1] - self.x[i]) / dx
                y_cur = y_cur + (self.y[i + 1] - self.y[i]) / dy
                i += 1

            # reduce length of arrays
            self.x = self.x[:i]
            self.y = self.y[:i]
            self.kx = self.kx[:i]
            self.ky = self.ky[:i]
