import os

import numpy as np
import scipy.interpolate as interp
from numpy.polynomial import Polynomial as poly
from scipy.optimize import minimize_scalar

from ufile import Ufile


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
        self.fpol = np.full((self.nx,), 2.2 * 0.55)
        self.pres = np.full((self.nx,), 5e3)
        self.ffprime = np.zeros((self.nx,))
        self.pprime = np.zeros((self.nx,))
        self.psi = np.zeros((self.nx, self.ny))  # so (x,y) axis order is in geqdsk format
        self.qpsi = np.zeros((self.nx,))
        self.rbdry = np.zeros((self.nx,))
        self.zbdry = np.zeros((self.nx,))
        self.rlim = np.zeros((self.nx,))
        self.zlim = np.zeros((self.nx,))

        if not (gd is None):
            for key in gd.keys():
                exec('self.' + key + ' = gd["' + key + '"]')
        self._values = np.ndarray((self.nx, 0))

        self._psi_interp = interp.RectBivariateSpline(np.linspace(self.rleft, self.rleft + self.rdim, self.nx),
                                                      np.linspace(self.zmid - self.zdim / 2, self.zmid + self.zdim / 2,
                                                                  self.ny),
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
        # internal n_theta
        grid_step = np.diff(self.r).min()
        theta = np.round(2 * np.pi * self.r.max() / grid_step).astype(np.int64)
        self._n_theta = min(RXYTE._n_theta_max, theta)

    def self_area(self, theta=None):
        # This method returns area of geometry as radii(columns) x thetas (rows) meshgrids
        # This method should be used to convert input data given in various ways to angles in radians
        # and to combine radii and angles in meshgrids
        if theta is None:
            theta = self._n_theta

        if isinstance(theta, (int, np.int64)):
            n_theta = theta
            theta = np.linspace(0, 2 * np.pi, n_theta + 1)[:-1]

        r, theta = np.meshgrid(self.r, theta)

        return r, theta

    def self_cartesian(self, theta=None):
        # This method returns cartesian coordinates of surfaces points as meshgrids where rows and columns correspond
        # to surface radius and poloidal angle
        # So thus points are not equidistant, columns and raws are not parallel, columns are not orthogonal to raws at large scales.
        # But local orthogonality is presented, which can be used to compute gradients.
        # In this case, we get a gradient in the form of poloidal and radial components.
        r, theta = self.self_area(theta=theta)
        xx = self.x + r * (np.cos(theta) - self.tria * np.sin(theta) ** 2)
        yy = self.y + r * self.elon * np.sin(theta)
        return xx, yy, r, theta

    def self_polar(self, theta=None):
        # This method is similar to self_cartesian. But returns geometry  polar coordinates. (Full Area)
        xx, yy, r, theta = self.self_cartesian(theta)
        vec = xx + 1j * yy
        return np.abs(vec), np.angle(vec), r, theta

    @property
    def nested(self):
        xx, yy, *_ = self.self_cartesian()
        inout = 1
        for ri in range(len(self.r) - 1):
            inout *= -check_inside((xx[:, ri] - self.x[ri], yy[:, ri] - self.y[ri]),
                                   (xx[:, ri + 1] - self.x[ri], yy[:, ri + 1] - self.y[ri]))
        if inout == 1:
            return True
        else:
            return False

    def r_lcs(self, limiter):
        xx, yy, *_ = self.self_cartesian()
        limiter = to_complex(limiter)
        ri = 0
        while ri < len(self.r) and \
                check_inside((xx[:, ri] - self.x[ri], yy[:, ri] - self.y[ri]),
                             (limiter.real - self.x[ri], limiter.imag - self.y[ri])):
            ri += 1
        if ri < len(self.r) - 1:
            rlcs, lcs = rlcs_from_params(shifx=lc(self.r[ri], self.r[ri + 1], self.x[ri], self.x[ri + 1]),
                                         shify=lc(self.r[ri], self.r[ri + 1], self.y[ri], self.y[ri + 1]),
                                         trian=lc(self.r[ri], self.r[ri + 1], self.tria[ri], self.tria[ri + 1]),
                                         elon=lc(self.r[ri], self.r[ri + 1], self.elon[ri], self.elon[ri + 1]),
                                         limiter=limiter)
        else:
            rlcs, lcs = rlcs_from_params(self.x[-1], self.y[-1], trian=self.tria[-1], elon=self.elon[-1],
                                         limiter=limiter)

        return rlcs, lcs

    def cartesian2r(self, points):
        # returns radii of surfaces for inputed points
        # Points should be given in same way as in scipy.interpolate.griddata
        xx, yy, r, theta = self.self_cartesian()
        return interp.griddata((xx.flatten(), yy.flatten()), r.flatten(), points)


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
                           options={'xatol': 1e-4, 'maxiter': 5000, 'disp': 0}).x
    return rlcs, lcs(rlcs)


def rxyte_from_params(shifx=0., shify=0., trian=0., elon=1., num=100, limiter=7.85, length_unit='cm'):
    rlcs, lcs = rlcs_from_params(shifx, shify, trian, elon, limiter)

    # convert inputs to polynomial objects
    shifx = poly(shifx)
    shify = poly(shify)
    trian = poly(trian)
    elon = poly(elon)

    rxyte = np.zeros((num, 5))
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


def read_astra_ufiles(folder):
    files = os.listdir(folder)
    tbl = dict()  # out table
    for file in files:
        ufile = Ufile(folder + '\\' + file)
        tbl[file] = ufile.y
    tbl['a'] = ufile.x

    return tbl


def read_astra_tab(filename):
    with open(filename, 'r') as fh:
        lines = list()
        for ln in fh.readlines():
            ln = ln.split()
            for ind, tok in enumerate(ln):
                try:
                    # TODO conversion from old fortran format(without e 1.123-5) to exponential (1.123e-5)
                    ln[ind] = float(tok)
                except ValueError:
                    pass
            lines.append(ln)

    tbl = dict()  # out table
    tbls = list()

    arr = np.ndarray((0,))
    names = list()
    for ln in lines:
        if not all([isinstance(tok, float) for tok in ln]):
            # try to store table
            if arr.size > 0:
                tbls.append({'names': names, 'values': arr})
                names = list()
                arr = np.ndarray((0,))

            # try extract name = value pairs
            toks = [format(_).split('=') for _ in ln]
            for ind, ele in enumerate(toks[:-1]):
                if len(ele) == 2:
                    if len(toks[ind + 1]) == 1:
                        if ele[1] == '':
                            ele[1] = toks.pop(ind + 1)[0]
                    try:
                        tbl[ele[0]] = float(ele[1])
                    except ValueError:
                        pass
            # renew names and arr for table
            if all([not (isinstance(tok, float)) for tok in ln]):
                names = ln
                arr = np.ndarray((0, len(names)))
        elif len(ln) == len(names):
            arr = np.vstack((arr, np.array(ln)))

    # ASTRA radial output tables should contain data with NA1 radial points
    # First table is time output
    if not all(np.diff([len(tb['values']) for tb in tbls[1:]]) == 0):
        print('Warning. Tables have different length')
        return tbls

    # collecting data
    for tb in tbls[:]:
        for ind, key in enumerate(tb['names'][:]):
            value = tb['values'][:, ind]
            if key in tbl:
                if not np.isscalar(tbl[key]):
                    if not all(value == tbl[key]):
                        print(' Warning. Different values with same name in tab')
                        suffix = 0
                        while key in tbl.keys():
                            key = key + format(suffix)
                            suffix += 1
            tbl[key] = value

    return tbl


def rxyte_from_astra_tab(tbl):
    # TODO change names of parameters in astra
    return RXYTE(tbl['a'], x=tbl['shif'], y=tbl['shiv'], tria=tbl['tria'], elon=tbl['elon'])


def geqdsk_from_astra_tab(tbl, nx=65, ny=65, rdim=0.25, zdim=0.25, rmain=0.55, btor=2.2, current=25000, boundary=None,
                          limiter=None, theta=360):
    if isinstance(tbl, str):
        try:
            tbl = read_astra_tab(tbl)
        except:
            tbl = None
    if tbl is None:
        return None
    if 'R' in tbl.keys():
        rmain = tbl['R']
    if 'B' in tbl.keys():
        btor = tbl['B']
    if 'I' in tbl.keys():
        current = tbl['I'] * 1e6

    geom = rxyte_from_astra_tab(tbl)
    geqdsk = dict()
    geqdsk['nx'] = nx
    geqdsk['ny'] = ny
    geqdsk['rdim'] = rdim
    geqdsk['zdim'] = zdim
    geqdsk['rcentr'] = rmain
    geqdsk['rleft'] = rmain - geqdsk['rdim'] / 2
    geqdsk['zmid'] = 0.
    geqdsk['rmagx'] = tbl['shif'][0] + rmain
    geqdsk['zmagx'] = tbl['shiv'][0]
    geqdsk['simagx'] = tbl['psi'][0] / 2 / np.pi
    geqdsk['sibdry'] = tbl['psi'][-1] / 2 / np.pi
    geqdsk['bcentr'] = btor
    geqdsk['cpasma'] = current

    xx, yy, rr, theta = geom.self_cartesian(theta=theta)

    """ interpolation from psi to value"""
    # equidistant psi vector from value at magnetic axis to boundary
    psi1 = np.linspace(tbl['psi'][0] / 2 / np.pi, tbl['psi'][-1] / 2 / np.pi, geqdsk['nx'])

    psi2fpol = interp.interp1d(tbl['psi'] / 2 / np.pi, tbl['IPOL'] * rmain * btor)
    geqdsk['fpol'] = psi2fpol(psi1)

    psi2pres = interp.interp1d(tbl['psi'] / 2 / np.pi, tbl['PRt'])
    geqdsk['pres'] = psi2pres(psi1) * 1e6

    psi2ff = interp.interp1d(tbl['psi'] / 2 / np.pi, tbl['ff'])
    geqdsk['ffprime'] = -psi2ff(psi1) * 1e6 * rmain  # may be 1e6 should be mu0

    psi2pf = interp.interp1d(tbl['psi'] / 2 / np.pi, tbl['pf'])
    geqdsk['pprime'] = -psi2pf(psi1) * 1e6 / rmain

    psi2q = interp.interp1d(tbl['psi'] / 2 / np.pi, tbl['q'])
    geqdsk['qpsi'] = psi2q(psi1)

    r = np.linspace(geqdsk['rleft'], geqdsk['rleft'] + geqdsk['rdim'], geqdsk['nx'])
    z = np.linspace(geqdsk['zmid'] - geqdsk['zdim'] / 2, geqdsk['zmid'] + geqdsk['zdim'] / 2, geqdsk['ny'])
    rz, zr = np.meshgrid(r, z)

    r2psi = interp.interp1d(tbl['a'], tbl['psi'] / 2 / np.pi, bounds_error=False,
                            fill_value=(tbl['psi'][0] / 2 / np.pi, 1.1 * tbl['psi'][-1] / 2 / np.pi))
    geqdsk['psi'] = interp.griddata((xx.flatten() + rmain, yy.flatten()), r2psi(rr.flatten()), (rz, zr),
                                    method='linear')
    geqdsk['psi'][np.isnan(geqdsk['psi'])] = geqdsk['sibdry'] * 1.1

    # write boundary
    geqdsk['rbdry'] = xx[:, -1] + rmain
    geqdsk['zbdry'] = yy[:, -1]

    # write limiter
    if np.isscalar(limiter):
        rlim = rmain + limiter * np.cos(theta[:, 0])
        zlim = limiter * np.sin(theta[:, 0])
    else:
        rlim = None

    if not (rlim is None):
        geqdsk['rlim'] = rlim
        geqdsk['zlim'] = zlim

    return geqdsk


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


def lc(x1, x2, y1, y2):
    return (x1 * y2 - x2 * y1) / (x1 - x2), (y1 - y2) / (x1 - x2)
