from scipy.interpolate import CubicSpline
from scipy.misc import derivative
import numpy as np


class pdata:
    def __init__(self, r=None, ph=None):
        if r != [] and ph != []:
            self.r = r
            self.ph = ph
        else:
            self.r, self.ph = (list() for _ in range(2))

    def endscalculate(self, indexes):
        """
            in case of non zero end calculating zero end by variational spline
        """

        success = 1

        phasespline = CubicSpline(self.r, self.ph, bc_type='natural')
        splinder = phasespline.derivative()
        out = splinder(self.r)
        firstder = out[1]
        lastder = out[-1]  # Edited by Vladislav

        if indexes[0] == len(self.r) or indexes[len(indexes)-1] == len(self.r):
            if lastder >= 0:
                success = 0
                return success
            k = lastder
            b = self.ph[-1] - k * self.r[-1]
            r = -b / k
            self.r.append(r)
            self.ph.append(0)
            #   Matlab: phasedata(length([phasedata.r])+1).r=r;
            #           phasedata(length([phasedata.r])).v=0;
        if indexes[0] == 0:
            if firstder <= 0:
                success = 0
                return success
            k = firstder
            b = self.ph[0] - k * self.r[0]
            r = -b / k      # why negative radii?
            data = pdata([r], [0])
            for i in range(len(self.r) + 1):
                data.r[i + 1].append(self.r[i])
                data.ph[i + 1].append(self.ph[i])
            del self.r, self.ph
            self.r = data.r
            self.ph = data.ph

        return success


class cake:
    def __init__(self, radius=None, center=None, thick=None):
        if not (radius or center or thick):
            self.radius, self.center, self.thick = (list() for _ in range(3))
        else:

            self.radius = radius
            self.center = center
            self.thick = thick

    def __add__(self, other):   #   for +=
        self.radius.append(other.radius)
        self.center.append(other.center)
        self.thick.append(other.thick)
        return self

    def pop(self):
        self.radius.pop()
        self.center.pop()
        self.thick.pop()

    def __len__(self):
        return len(self.radius)


class point:
    def __init__(self, x=None, fx=None):
        self.x = x
        self.fx = fx

    def __len__(self):
        return len(self.x)




