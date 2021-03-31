import numpy as np
from scipy import constants as const
from sidtrace import trace


# from sidtrace import check


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

            # convert to grid coordinates
            self.x[0] = n_field.x_interpolant(self.x[0])
            self.y[0] = n_field.y_interpolant(self.y[0])

            # trace with precompiled code
            i = trace(self.x, self.y, self.kx, self.ky,
                      n_field.st, n_field.dx[0], n_field.dy[0],
                      n_field.nx, n_field.ny,
                      n_field.ax, n_field.ay)

            # reduce length of arrays
            self.x = self.x[:i]
            self.y = self.y[:i]
            self.kx = self.kx[:i]
            self.ky = self.ky[:i]

            # convert to physical values
            self.x = x0 + (self.x - self.x[0]) * n_field.dx[0]
            self.y = y0 + (self.y - self.y[0]) * n_field.dy[0]