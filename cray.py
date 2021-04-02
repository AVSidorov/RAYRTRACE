import numpy as np
from scipy import constants as const
from sidtrace import trace
# from sidtrace import check


class Ray:
    MaxStep = 100000  # Settings variable for arrays preallocating
    kstep = 1

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
        self.n_field = n_field

        # launch ray if density is given
        if not (n_field is None):
            # change wave vector corresponding to local density
            refr = np.sqrt(1 - n_field.n_interpolant(self.x[0], self.y[0]) / n_field.nc)
            self.kx[0] = self.kx[0] * refr
            self.ky[0] = self.ky[0] * refr

            # convert to grid coordinates
            self.x[0] = n_field.x2grid(self.x[0])
            self.y[0] = n_field.y2grid(self.y[0])

            st = n_field.st * Ray.kstep
            # trace with precompiled code
            i = trace(self.x, self.y, self.kx, self.ky,
                      st, n_field.dx[0], n_field.dy[0],
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

            # store steps
            # Step can be changed, if points will be added
            self.st = np.full_like(self.x, st)
            self.st[-1] = 0  # step is bound with segment start point

    @property
    def k(self):
        return np.sqrt(self.kx ** 2 + self.ky ** 2)

    @property
    def lax(self):
        kavr = (self.k[:-1] + self.k[1:]) / 2
        return np.cumsum(np.insert(self.st[:-1] * kavr, 0, 0))

    @property
    def ph(self):
        kavr = (self.k[:-1] + self.k[1:]) / 2
        dl = np.diff(self.lax)
        return np.cumsum(np.insert(dl * kavr, 0, 0))
        #  dPh = dl * (k1+k0)/2 exact value of integral if linear expansion of k is used
        # if k = k0 + a*dl the analytical integration gives relation st = dl * (log k1 - log k0)/(k1-k0)
        # on small steps (k1+k0)/2 is very close to (log k1 - log k0)/(k1-k0)

    def addpoint(self, val, axis='y'):
        # axis determination
        y = self.y
        ky1 = self.ky[-1]

        if axis == 'x':
            y = self.x
            ky1 = self.kx[-1]

        # Search existing and check ability to insert
        if val in y:
            return np.array(np.where(y == val)).flatten()
        elif np.any(y < val) and np.any(val < y):
            # interval number
            i_n = np.sort(np.array(np.where((y - val)[:-1] * (y - val)[1:] < 0)).flatten())
        elif (val - y[-1]) * ky1 > 0:
            i_n = np.array(len(y) - 1).flatten()
        else:
            return np.nan

        # Get data for segments in which will be inserted points

        # start segment point
        x0 = self.x[i_n]
        y0 = self.y[i_n]

        kx0 = self.kx[i_n]
        ky0 = self.ky[i_n]

        st0 = self.st[i_n]

        # grid indexes
        x_i = np.array(np.trunc(self.n_field.x2grid(x0)), dtype=int)
        y_i = np.array(np.trunc(self.n_field.y2grid(y0)), dtype=int)

        ax0 = self.n_field.ax[y_i, x_i]
        ay0 = self.n_field.ay[y_i, x_i]

        # POINTS INSERTING
        ind_inserted = np.ndarray((0,), dtype=int)
        # variable to shift indexes after inserting
        i_n_sh = 0

        for xx, yy, ax, ay, kx, ky, st1, ind in zip(x0, y0, ax0, ay0, kx0, ky0, st0, i_n):
            st = st1
            if axis == 'y':
                if ay != 0:
                    st = np.array(((-ky - np.sqrt(ky ** 2 + 2 * ay * (val - yy))) / ay,
                                   (-ky + np.sqrt(ky ** 2 + 2 * ay * (val - yy))) / ay))
                    st = st[st >= 0].min()
                    if st == 0 and val != yy and ky != 0:  # accuracy at small ay is bad so non zero step can be lost
                        st = (val - yy) / ky
                elif ky != 0:
                    st = (val - yy) / ky
            elif axis == 'x':
                if ax != 0:
                    st = np.array(((-kx - np.sqrt(kx ** 2 + 2 * ax * (val - xx))) / ax,
                                   (-kx + np.sqrt(kx ** 2 + 2 * ax * (val - xx))) / ax))
                    st = st[st >= 0].min()
                    if st == 0 and val != xx and kx != 0:  # accuracy at small ax is bad. So non zero step can be lost
                        st = (val - xx) / kx
                elif kx != 0:
                    st = (val - xx) / kx

            # st = st[st >= 0].min()
            # at least one zero or positive root must be
            # since
            # interval number is found
            # that means in this segment of ray end points are so "start" <= val <= "end"
            # so case then corresponding  kx(or ky) == 0 and ax(or ay) == 0 is impossible
            # or
            # ray propagates in direction of new value and kx(or ky) != 0
            # st == 0 means coincidence given value with existing point in ray

            if st != 0:
                ind_inserted = np.append(ind_inserted, ind + i_n_sh + 1)
                if st != st1:
                    self.st[ind + i_n_sh] = st

                    if ind + i_n_sh + 1 != len(self.x):
                        self.st = np.insert(self.st, ind + i_n_sh + 1, st1 - st)
                        # TODO correct step determination as above it seems new function is necessary
                    else:
                        self.st = np.insert(self.st, ind + i_n_sh + 1, 0)

                    self.kx = np.insert(self.kx, ind + i_n_sh + 1, kx + ax * st)
                    self.ky = np.insert(self.ky, ind + i_n_sh + 1, ky + ay * st)

                    self.x = np.insert(self.x, ind + i_n_sh + 1, xx + kx * st + ax / 2 * st ** 2)
                    self.y = np.insert(self.y, ind + i_n_sh + 1, yy + ky * st + ay / 2 * st ** 2)
                    # after inserting interval nums change
                    i_n_sh += 1

        return ind_inserted
