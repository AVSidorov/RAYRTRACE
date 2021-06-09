import numpy as np
from numpy.polynomial import Polynomial as poly
import scipy.interpolate as interp
from scipy.optimize import minimize_scalar


class Equilibrium:
    """
    class represents tokamak equilibrium in geqdsk format like form
    """
    def __init__(self, gd=None, tokamak=None):
        self.nx = 65
        self.ny = 65
        self.rdim = 0.2
        self.zdim = 0.2
        self.rcentr = 0.55
        self.rleft = 0.45
        self.zmid = 0.
        self.rmagx = 0.55
        self.zmagx = 0.
        self.simagx = 0.
        self.sibdry = 0.
        self.bcentr = 2.2
        self.cpasma = 22e3
        self.fpol = np.full((self.nx,), 2.2*0.55)
        self.pres = np.full((self.nx,), 5e3)
        self.ffprime = np.zeros((self.nx,))
        self.pprime = np.zeros((self.nx,))
        self.psi = np.zeros((self.nx, self.ny))  # so (x,y) axis order is in geqdsk format
        self.qpsi = np.zeros((self.nx,))
        self.rbdry = np.zeros((self.nx,))
        self.zbdry = np.zeros((self.nx,))
        self.rlim = np.zeros((self.nx,))
        self.zlim = np.zeros((self.nx,))

        if not(gd is None):
            for key in gd.keys():
                exec('self.'+key+' = gd["'+key+'"]')
        self._values = np.ndarray((self.nx, 0))

        self._psi_interp = interp.RectBivariateSpline(np.linspace(self.rleft, self.rleft+self.rdim, self.nx),
                                                      np.linspace(self.zmid-self.zdim/2, self.zmid+self.zdim/2, self.ny),
                                                      self.psi, kx=1, ky=1)


class RXYTE:
    """
        Class represents geometry of psi through shifts, elongation and triangulation
    """
    _n_theta_max = 1000

    def __init__(self, r, x=None, y=None, tria=None, elon=None):
        if isinstance(r, np.ndarray):
            if r.ndim == 2:
                if r.shape[1] >= 5:
                    r, x, y, tria, elon = (r[:, i] for i in range(5))
        elif isinstance(r, (int, np.int64)):
            (self.r, self.x, self.y, self.tria) = (np.zeros((r,)) for _ in range(4))
            self.elon = np.ones((r, 1))
        elif isinstance(r, (tuple, list)):
            if len(r) == 2:
                self.r = np.linspace(0, r[0], r[1])
                (self.x, self.y, self.tria) = (np.zeros((r[1],)) for _ in range(3))
                self.elon = np.ones((r[1], 1))
        ind = np.argsort(r)
        self.r = r[ind]
        self.x = x[ind]
        self.y = y[ind]
        self.tria = tria[ind]
        self.elon = elon[ind]

    def self_polar(self, theta=None):

        if theta is None:
            grid_step = np.diff(self.r).min()
            theta = np.round(2 * np.pi * self.r.max() / grid_step).astype(np.int64)
            theta = min(RXYTE._n_theta_max, theta)

        if isinstance(theta, (int, np.int64)):
            n_theta = theta
            theta = np.linspace(0, 2 * np.pi, n_theta + 1)[:-1]

        r, theta = np.meshgrid(self.r, theta)

        return r, theta

    def self_cartesian(self, theta=None):
        r, theta = self.self_polar(theta=theta)
        xx = self.x + r * (np.cos(theta) - self.tria * np.sin(theta)**2)
        yy = self.y + r * self.elon * np.sin(theta)
        return xx, yy

    @property
    def nested(self):
        xx, yy = self.self_cartesian()
        inout = 1
        for ri in range(len(self.r) - 1):
            inout *= -check_inside((xx[:, ri] - self.x[ri], yy[:, ri] - self.y[ri]),
                                   (xx[:, ri + 1] - self.x[ri], yy[:, ri + 1] - self.y[ri]))
        if inout == 1:
            return True
        else:
            return False

    def r_lcs(self, limiter):
        xx, yy = self.self_cartesian()
        limiter = to_complex(limiter)
        ri = 0
        while ri<len(self.r) and \
              check_inside((xx[:, ri]-self.x[ri], yy[:, ri] - self.y[ri]),
                           (limiter.real-self.x[ri], limiter.imag-self.y[ri])):
            ri += 1
        if ri < len(self.r)-1:
            rlcs, lcs = rlcs_from_params(shifx=lc(self.r[ri], self.r[ri+1], self.x[ri], self.x[ri+1]),
                                         shify=lc(self.r[ri], self.r[ri+1], self.y[ri], self.y[ri+1]),
                                         trian=lc(self.r[ri], self.r[ri+1], self.tria[ri], self.tria[ri+1]),
                                         elon=lc(self.r[ri], self.r[ri + 1], self.elon[ri], self.elon[ri + 1]),
                                         limiter=limiter)
        else:
            rlcs, lcs = rlcs_from_params(self.x[-1], self.y[-1], trian=self.tria[-1], elon=self.elon[-1], limiter=limiter)

        return rlcs, lcs

def read_astra_tab(filename):
    with open(filename,'r') as fh:
        lines = list()
        for ln in fh.readlines():
            ln = ln.split()
            for ind, tok in enumerate(ln):
                try:
                    ln[ind] = float(tok)
                except ValueError:
                    pass
            lines.append(ln)

    tbls = list()
    arr = np.ndarray((0,))
    for ln in lines:
        if not all([isinstance(tok, float) for tok in ln]):
            if arr.size > 0:
                tbls.append({'names': names, 'values': arr})
            names = ln
            arr = np.ndarray((0, len(names)))
        elif len(ln) == len(names):
            arr = np.vstack((arr, np.array(ln)))

    # ASTRA radial output tables should contain data with NA1 radial points
    # First table is time output
    if not all(np.diff([len(tb['values']) for tb in tbls[1:]]) == 0):
        print( 'Warning. Tables have different length')
        return tbls

    # collecting data
    tbl = dict()
    tbl[tbls[1]['names'][0]]=tbls[1]['values'][:, 0]

    for tb in tbls[1:]:
        for ind, key in enumerate(tb['names'][1:]):     # first column is a radius of magnetic surface
            value = tb['values'][:, ind+1]
            if key in tbl:
                if all(value == tbl[key]):
                    continue
                else:
                    print(' Warning. Different values with same name in tab')
                    suffix = 0
                    while key in tbl.keys():
                        key = key+format(suffix)
                        suffix += 1
            tbl[key] = value

    return tbl


def rlcs_from_params(shifx=0., shify=0., trian=0., elon=1., limiter=7.85):

    limiter = to_complex(limiter)

    shifx = poly(shifx)
    shify = poly(shify)
    trian = poly(trian)
    elon = poly(elon)

    theta = np.angle(limiter)
    lcs = lambda r: (shifx(r) + r * (np.cos(theta) - trian(r) * np.sin(theta) ** 2)) + \
                    1j * (shify(r) + r * elon(r) * np.sin(theta))

    def dist_from_limiter(lmt, lcs1):
        if all(np.abs(lcs1) <= np.abs(lmt)):
            return (np.abs(lmt) - np.abs(lcs1)).min()
        else:
            return np.inf

    rlcs = minimize_scalar(lambda r: dist_from_limiter(limiter, lcs(r)),
                           bracket=(0, np.abs(limiter).max()),
                           bounds=(0, np.abs(limiter).max()),
                           args=(), method='bounded', tol=None,
                           options={'xatol': 1e-4, 'maxiter': 5000, 'disp':0}).x
    return rlcs, lcs(rlcs)

def rxyte_from_params(shifx=0., shify=0., trian=0., elon=1., num=100, limiter=7.85, length_unit='cm'):

    rlcs, lcs = rlcs_from_params(shifx, shify, trian, elon, limiter)

    # convert inputs to polynomial objects
    shifx = poly(shifx)
    shify = poly(shify)
    trian = poly(trian)
    elon = poly(elon)

    rxyte = np.zeros((num,5))
    rxyte[:, 0] = np.linspace(0, rlcs, num)
    rxyte[0, 0] = 1e-4
    rxyte[:, 1] = shifx(rxyte[:, 0])
    rxyte[:, 2] = shify(rxyte[:, 0])
    rxyte[:, 3] = trian(rxyte[:, 0])
    rxyte[:, 4] = elon(rxyte[:, 0])

    return RXYTE(rxyte), rlcs, lcs


def check_inside(a, b):
    # for this algorithm origin must be inside of convex contour
    # for working in polar coordinates make complex numbers
    a = to_complex(a)
    b = to_complex(b)

    ang2rho = interp.interp1d(np.angle(b), np.abs(b), kind='cubic', bounds_error=False,
                       fill_value=(np.abs(b[0]), np.abs(b[-1])))

    # radius coordinates of "b" on same angels as in "a"
    rb = ang2rho(np.angle(a))

    if all(np.abs(a) < rb):
        return -1  # a is inside b
    elif all(np.abs(a) > rb):
        return 1  # a is outside b
    else:
        return 0


def rxyte_from_astra_tab(tbl):
    # TODO change names of parameters in astra
    return RXYTE(tbl['a'], x=tbl['shif'], y=tbl['shiv'], tria=tbl['tria'], elon=tbl['elon'])


def geqdsk_from_astra_tab(tbl, rmain=0.55, rdia=0.0785):
    geom = rxyte_from_astra_tab(tbl)
    geqdsk = dict()
    geqdsk['nx'] = 65
    geqdsk['ny'] = 65
    geqdsk['rdim'] = 0.2
    geqdsk['zdim'] = 0.2
    geqdsk['rcentr'] = rmain
    geqdsk['rleft'] = rmain - geqdsk['rdim']/2
    geqdsk['zmid'] = 0.
    geqdsk['rmagx'] = geom.psi2x(0) + rmain
    geqdsk['zmagx'] = geom.psi2y(0)
    geqdsk['simagx'] = geom.psi.min()


def to_complex(curve):
    if np.isscalar(curve):
        theta = np.linspace(0, 2 * np.pi, RXYTE._n_theta_max, endpoint=False)
        curve = curve * np.cos(theta) + 1j * curve * np.sin(theta)
    elif isinstance(curve, (tuple, list)):
        if len(curve) == 2:
            curve = curve[0] + 1j * curve[1]
    elif isinstance(curve, np.ndarray):
        if curve.ndim == 2:
            curve = curve[:, 0] + 1j * curve[:, 1]
    curve = curve[np.argsort(np.angle(curve))]
    return curve


def lc(x1,x2,y1,y2):
    return (x1*y2-x2*y1)/(x1-x2), (y1-y2)/(x1-x2)