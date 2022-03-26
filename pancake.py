import numpy as np
from scipy.interpolate import interp1d

import ft2
from den import phase2den, den2phase


def phases_add_ends(ph, n=10, r_lcs=ft2.r_dia, shift=0, bkg=1e12, r_bkg=ft2.r_ant):
    ph = ph.copy()
    x = np.concatenate((np.linspace(-r_bkg, shift - r_lcs, n), np.linspace(shift + r_lcs, r_bkg, n)))
    l_chords = chord_length(x, r_bkg)

    if ph.shape[1] == 2:
        np.column_stack((ph[:, 0], ph[:, 1], ph[:, 1] * 0.1))

    dph_dl = den2phase(bkg)
    ph_add = np.column_stack((x, l_chords * dph_dl, np.zeros_like(x)))
    ph = np.append(ph, ph_add, axis=0)
    ph = ph[ph[:, 0].argsort()]

    return ph


def phase_spline(ph, x=None, x_err=1.):
    if isinstance(x, int):
        x = np.linspace(ph[:, 0].min(), ph[:, 0].max(), x)

    err_ph = np.random.randn(ph.shape[0]) * ph[:, 2] / 6
    while not all(np.abs(err_ph) <= ph[:, 2]):
        err_ph = np.random.randn(ph.shape[0]) * ph[:, 2] / 6

    err_x = np.random.randn(ph.shape[0]) * np.sign(ph[:, 2]) * x_err / 6
    while not all(np.abs(err_x) <= np.sign(ph[:, 2]) * x_err):
        err_x = np.random.randn(ph.shape[0]) * np.sign(ph[:, 2]) * x_err / 6

    phs = interp1d(ph[:, 0] + err_x, ph[:, 1] + err_ph, kind='cubic', bounds_error=False, fill_value=0)

    if x is not None:
        return phs(x), x
    else:
        return phs


def pancake(x, ph, bkg=0, elong=None, trian=None):
    # x must be sorted
    ind = np.argsort(x)
    x = x[ind]
    ph = ph[ind]

    # add zeros to ends of ph
    x = np.insert(x, 0, x.min() - 0.1)
    x = np.append(x, x.max() + 0.1)
    ph = np.insert(ph, 0, 0)
    ph = np.append(ph, 0)

    radii, shift, dPhdL = (np.ndarray((0,)) for _ in range(3))
    while any(ph != 0) and x.size > 3:
        x, ph, r, c, dph = eat_pancake(x, ph, elong, trian)
        radii, shift, dPhdL = (np.append(ele, val) for ele, val in zip((radii, shift, dPhdL), (r, c, dph)))

    dPhdL = np.cumsum(dPhdL)
    ind = np.argsort(radii)
    radii, shift, dPhdL = (ele[ind] for ele in (radii, shift, dPhdL))

    return radii, shift, dPhdL


def eat_pancake(x, ph, elong=None, trian=None):
    # x must be sorted
    ph[[0, -1]] = 0

    # cut leading and trailing zeros
    indSt = np.where(ph != 0)[0].min()
    indEn = np.where(ph != 0)[0].max()

    x = x[np.max((0, indSt - 1)):np.min((indEn + 2, x.size))]
    ph = ph[np.max((0, indSt - 1)):np.min((indEn + 2, ph.size))]

    if x.size < 1:
        return x, ph, 0, np.nan, 0

    r = (x[-1] - x[0]) / 2
    c = (x[-1] + x[0]) / 2

    if x.size < 3:
        return x, ph, r, c, 0

    l_chords = chord_length(x - c, r, elong, trian)
    l_chords[[0, -1]] = 0

    thickness = np.zeros_like(ph)
    thickness[l_chords != 0] = ph[l_chords != 0] / l_chords[l_chords != 0]

    # choose head or tail
    end_ind = np.abs(thickness[[1, -2]]).argmin()
    dPhdL = thickness[[1, -2]][end_ind]
    # new ph "eat pancake"
    ph = ph - l_chords * dPhdL

    # exact 0 at chosen end
    ph[np.array([1, -2])[end_ind]] = 0

    # add new point (next pancake end)
    # find zero cross points position x0
    dy = np.diff(ph[[1, 2, -3, -2]])[[0, -1]]
    y1, y2 = ph[[1, -3]], ph[[2, -2]]
    x1, x2 = x[[1, -3]], x[[2, -2]]
    x0 = x[[1, -2]]
    x0[dy != 0] = (x1 * y2 - x2 * y1)[dy != 0] / dy[dy != 0]

    # shift potential new ends x[[1,-2]]
    # potential so as in case constant density we can obtain several leading and trailing zeros

    # shift only points are inside last pancake
    mask = np.array([x0[0] - x[0], x[-1] - x0[1]]) > 0
    x[np.array([1, -2])[mask]] = x0[mask]
    # make exact zero in shifted points
    ph[np.array([1, -2])[mask]] = 0

    return x, ph, r, c, dPhdL


def chord_length(x, r, elong=None, trian=None):
    # calculates vertical chord length at x grid
    # for surface with radius r given triangularity and elongation
    # It assumes center of surface at x=0

    if elong is None:
        elong = 1
    if trian is None:
        trian = 0

    x = np.atleast_1d(x)
    x_in = x.copy()
    mask = np.abs(x) < r
    x = x[mask]
    if trian != 0:
        # solve quadratic equation relative cos
        cos = (1 - np.sqrt(1 + 4 * trian * (x / r + trian))) / 2 / trian
    else:
        cos = x / r

    sin = np.sqrt(1 - cos ** 2)

    y = r * elong * sin

    length = np.zeros_like(x_in)
    length[mask] = 2 * y
    return length
