import numpy as np
from scipy import constants as const


class ray:
    def __init__(self, x0=0, y0=0, kx0=0, ky0=1, f=135e9):
        self.x = np.array([x0, ])
        self.y = np.array([y0, ])
        self.l = np.array([0, ])
        self.freq = f
        self.w = 2 * np.pi * f
        # if freq is not set try to get frequency from wave number
        if np.sqrt(kx0 ** 2 + ky0 ** 2) == 1:
            # assume that only initial direction of ray is given
            kx0 = kx0 * self.w / const.speed_of_light
            ky0 = ky0 * self.w / const.speed_of_light
        elif self.freq == 135e9:
            # if default value of frequency is used
            self.w = np.sqrt(kx0 ** 2 + ky0 ** 2) * const.speed_of_light
            self.freq = self.w / 2 / np.pi

        self.kx = np.array([kx0, ])
        self.ky = np.array([ky0, ])
