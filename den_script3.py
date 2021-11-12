import h5py
import numpy as np

test_run = True

# Tokamak Settings
r_dia = 7.85
r_ant = 10

# Date/Time/Discharge Settings
Ph = np.loadtxt('D:\\!SCN\\Density\\210708\\PH31_5.txt')
bkg = 5e12
ph_col = 1

TS = np.loadtxt('D:\\!SCN\\Density\\210708\\TS31_5.txt')
ts_den_col = -2
ts_den_err_col = -1
ts_te_col = 1
ts_te_err_col = 2

# Experimental data processing settings
rbf_eps = 1.
rbf_smth = 0.5
rbf_smth_t = 0.1

# Geometry preset
x00 = 1.
y00 = 0.
delta00 = 0.25
trian00 = 0.
elong00 = 1.075

x0_scale = 0.2
y0_scale = 0.01

sh_scale = 0.05
el_scale = 0.05
tr_scale = 0.01

# Output and control filenames
filename_res = 'MCres'
filename_stp = 'stop'
filename_ts = 'None'

import den as dn
from beam import Beam, field_result

Beam._ampbnd = 0.01
Beam._nray = 251

from scipy.interpolate import Rbf

import timeit
import os

if test_run >= 0:
    import matplotlib.pyplot as plt
    from matplotlib.pyplot import plot
    import plotfield as pf

# array for timing
t = np.ndarray((0,))

# lists for beams
beamsTX = list()
beamsRX = list()

x, y = (np.linspace(-12, 12, 1000) for _ in range(2))
xy, yx = np.meshgrid(x, y)

ant = np.array([x for x in np.linspace(-6.9, 6.9, 7)])
ant = np.vstack((ant, np.sqrt(r_ant ** 2 - ant ** 2)))
ant = ant.T

""" Vac Phases"""
nfield = dn.Nfield(np.zeros_like(xy), x=x, y=y)

for anten in ant:
    beamsTX.append(Beam(x0=anten[0], y0=-anten[1], kx0=0, ky0=1, n_field=nfield))
    beamsRX.append(Beam(x0=anten[0], y0=anten[1], kx0=0, ky0=-1, n_field=None))

ph_vac_full = np.ndarray((0,))
for beam_tx, beam_rx in zip(beamsTX, beamsRX):
    _, _, _, ph_max, *_ = field_result(beam_tx, beam_rx)
    ph_vac_full = np.append(ph_vac_full, ph_max)

for beams in (beamsTX, beamsRX):
    beams.clear()

del nfield, ph_max, beam_tx, beam_rx

""" Experimental data"""
yn = None
kden = None

# Pre prepared
if os.path.exists(filename_ts + '.h5'):
    with h5py.File(filename_ts + '.h5', 'r') as hf:
        if all([key in hf.keys() for key in ['yn', 'ny']]):
            yn = np.array(hf['yn'])
            ny = np.array(hf['ny'])

        if 'kden' in hf.keys():
            kden = np.array(hf['kden'])
        else:
            kden = None

yl = np.concatenate((TS[:, 0], y00 - TS[:, 0]))
nl = np.concatenate((TS[:, ts_den_col], TS[:, ts_den_col]))
if not (ts_den_err_col is None):
    nerrl = np.concatenate((TS[:, -1], TS[:, -1]))
else:
    nerrl = nl * 0.1

ind = np.logical_not(np.isnan(nl))
yl = yl[ind]
nl = nl[ind]
nerrl = nerrl[ind]

ind = np.argsort(yl)
yl = yl[ind]
nl = nl[ind]
nerrl = nerrl[ind]

if yn is None:
    rbfi = Rbf(yl, nl, epsilon=rbf_eps, smooth=rbf_smth)
    yn = np.linspace(yl.min(), yl.max(), 111)
    ny = rbfi(yn)
else:
    rbfi = None

if kden is None:
    ny, kden = dn.adjust2phase((Ph[:, 0], Ph[:, ph_col]), (1.5, yn, ny))

yl = np.insert(yl, 0, (-np.sqrt(r_ant ** 2 - 1.5 ** 2), -np.sqrt(r_dia ** 2 - 1.5 ** 2)))
yl = np.append(yl, (np.sqrt(r_ant ** 2 - 1.5 ** 2), np.sqrt(r_dia ** 2 - 1.5 ** 2)))

nl = np.insert(nl, 0, (bkg / kden, bkg / kden))
nl = np.append(nl, (bkg / kden, bkg / kden))

nerrl = np.insert(nerrl, 0, (0.1 * bkg / kden, 0.1 * bkg / kden))
nerrl = np.append(nerrl, (0.1 * bkg / kden, 0.1 * bkg / kden))

if not (rbfi is None):
    rbfi = Rbf(yl, nl, epsilon=rbf_eps, smooth=rbf_smth)
    yn = np.linspace(yl.min(), yl.max(), 111)
    ny = rbfi(yn)
    ny, kden = dn.adjust2phase((Ph[:, 0], Ph[:, ph_col]), (1.5, yn, ny))

ylt = np.concatenate((TS[:, 0], -TS[:, 0]))
tl = np.concatenate((TS[:, ts_te_col], TS[:, ts_te_col]))

if not (ts_te_err_col is None):
    terrl = np.concatenate((TS[:, ts_te_err_col], TS[:, ts_te_err_col]))
else:
    terrl = tl * 0.05

terrl = np.concatenate((TS[:, 2], TS[:, 2]))

ind = np.logical_not(np.isnan(tl))
ylt = ylt[ind]
tl = tl[ind]
terrl = terrl[ind]

ind = np.argsort(ylt)
ylt = ylt[ind]
tl = tl[ind]
terrl = terrl[ind]

rbfi = Rbf(ylt, tl, epsilon=rbf_eps, smooth=rbf_smth_t)
yt = np.linspace(ylt.min(), ylt.max(), 111)
ty = rbfi(yt)

del rbfi, ind

""" Create hdf5 file for storage results """
new = True
if os.path.exists(filename_res + '.h5'):
    new = False
    with h5py.File(filename_res + '.h5', 'r+') as hf:
        if not all([nms in hf.keys() for nms in ['x0', 'y0', 'delta', 'trian', 'elong', 'khi', 'rlcs']]):
            new = True
        elif any([hf[key].ndim != 2 for key in ['delta', 'trian', 'elong']]):
            new = True
        elif any([hf[key].shape[1] != 5 for key in ['delta', 'trian', 'elong']]):
            new = True
        hf.close()

    if new:
        fn = 0
        while os.path.exists(filename_res + '_' + format(fn, '03d') + '.h5'):
            fn += 1
        os.rename(filename_res + '.h5', filename_res + '_' + format(fn, '03d') + '.h5')
        del fn
if new:
    with h5py.File(filename_res + '.h5', 'w') as hf:
        hf.create_dataset('khi', shape=(0,), maxshape=(None,), dtype=float, compression='gzip', compression_opts=9)
        hf.create_dataset('rlcs', shape=(0,), maxshape=(None,), dtype=float, compression='gzip', compression_opts=9)
        hf.create_dataset('x0', shape=(0,), maxshape=(None,), dtype=float, compression='gzip', compression_opts=9)
        hf.create_dataset('y0', shape=(0,), maxshape=(None,), dtype=float, compression='gzip', compression_opts=9)
        hf.create_dataset('delta', shape=(0, 5), maxshape=(None, 5), dtype=float, compression='gzip',
                          compression_opts=9)
        hf.create_dataset('trian', shape=(0, 5), maxshape=(None, 5), dtype=float, compression='gzip',
                          compression_opts=9)
        hf.create_dataset('elong', shape=(0, 5), maxshape=(None, 5), dtype=float, compression='gzip',
                          compression_opts=9)
        hf.close()

del hf, new

rng = np.random.default_rng()

while not os.path.exists(filename_stp):
    test_run -= 1
    if not test_run:
        fl = open(filename_stp, 'w')
        fl.close()
        del fl
    """ Make by geometry parameters"""
    t = np.append(t, timeit.default_timer())

    x0 = rng.normal(loc=x00, scale=x0_scale, size=1)
    y0 = rng.normal(loc=y00, scale=y0_scale, size=1)

    delta = rng.normal(loc=delta00, scale=sh_scale, size=1)
    trian = rng.normal(loc=trian00, scale=tr_scale, size=1)
    elong = rng.normal(loc=elong00, scale=el_scale, size=1)
    r_obj, rlcs, _ = dn.rxyten_from_params(x0, 0, delta=0, trian=trian, elon=elong, rdia=r_dia)

    delta = np.array([delta[0], 0, -delta[0] / rlcs ** 2])
    trian = np.array([0, 0, trian[0] / rlcs ** 2])
    elon0 = 1.1 * elong
    while elon0 > elong:
        elon0 = rng.normal(loc=elong - min(0.1, elong00 - 1.), scale=0.01, size=1)

    elong = np.array([elon0[0], 0, (elong[0] - elon0[0]) / rlcs ** 2])
    r_obj, rlcs, _ = dn.rxyten_from_params(x0, y0, delta=delta, trian=trian, elon=elong, rdia=r_dia)

    if not r_obj.nested:
        print('skip')
        continue

    r_obj.map_on(x=1.5, y=yn, den=ny)

    t_obj, *_ = dn.rxyten_from_params(x0, y0, delta=delta, trian=trian, elon=elong, rdia=r_dia)
    t_obj.map_on(x=1.5, y=yt, den=ty)

    t = np.append(t, timeit.default_timer())
    print("RXYTEN create and map {:4.2f} s".format(t[-1] - t[-2]))

    """ Ray tracing """

    t = np.append(t, timeit.default_timer())
    den, _, _ = r_obj.on_grid(x=xy, y=yx)

    den[np.isnan(den)] = den[np.logical_not(np.isnan(den))].min()
    t = np.append(t, timeit.default_timer())
    print("Make grid {:4.2f} s".format(t[-1] - t[-2]))

    t = np.append(t, timeit.default_timer())
    nfield = dn.Nfield(den, x=x, y=y)
    t = np.append(t, timeit.default_timer())
    print("Make nfield {:4.2f} s".format(t[-1] - t[-2]))

    t = np.append(t, timeit.default_timer())
    for anten in ant:
        beamsTX.append(Beam(x0=anten[0], y0=-anten[1], kx0=0, ky0=1, n_field=nfield))
        beamsRX.append(Beam(x0=anten[0], y0=anten[1], kx0=0, ky0=-1, n_field=None))
    t = np.append(t, timeit.default_timer())
    print("Add beams {:4.2f} s".format(t[-1] - t[-2]))

    t = np.append(t, timeit.default_timer())

    ph_end_full = np.ndarray((0,))

    for beam_tx, beam_rx in zip(beamsTX, beamsRX):
        _, _, _, ph_max, _, _ = field_result(beam_tx, beam_rx)
        ph_end_full = np.append(ph_end_full, ph_max)

    # Clean up beams and density objects
    for beams in (beamsTX, beamsRX):
        beams.clear()

    t = np.append(t, timeit.default_timer())
    print("Get result phases {:4.2f} s".format(t[-1] - t[-2]))

    if test_run >= 0:
        theta = np.linspace(0, 2 * np.pi, 360, endpoint=False)
        _, xrt, yrt = r_obj.on_self_cartesian(theta)
        ri = np.where(r_obj.r >= rlcs)[0][0]

        plt.figure('Geometry')
        plt.imshow(den, cmap=plt.cm.jet, origin='lower', extent=[x.min(), x.max(), y.min(), y.max()])
        plot(xrt[:, ::int(xrt.shape[1] / 10)], yrt[:, ::int(xrt.shape[1] / 10)], lw=1)

        plot(r_dia * np.cos(theta), r_dia * np.sin(theta), 'k', lw=3)
        plot(xrt[:, ri], yrt[:, ri], 'r', lw=2)

        plt.colorbar()

        plt.figure('Phases')
        plt.grid(1)
        plt.xlabel('x, cm')
        plt.ylabel(r' Phase, $2\pi$')
        plt.title('Phases')
        plt.errorbar(Ph[:, 0], Ph[:, ph_col], Ph[:, ph_col] * 0.05, np.full((len(Ph, ),), 0.25), '.')
        plot(Ph[:, 0], (ph_vac_full - ph_end_full[:]) / 2 / np.pi)

        pf.plot_rxyten(r_obj)
        if not (ts_den_err_col is None):
            plt.errorbar(TS[:, 0], TS[:, ts_den_col] * kden, TS[:, ts_den_err_col] * kden, fmt='.', capsize=2, lw=2)
        else:
            plt.errorbar(TS[:, 0], TS[:, ts_den_col] * kden, TS[:, ts_den_col] * 0.1 * kden, fmt='.', capsize=2, lw=2)
        plot(yn, ny)
        plot(yl, nl * kden, '.')

        pf.plot_rxyten(t_obj, fig=plt.figure('RXYTET'))
        if not (ts_te_err_col is None):
            plt.errorbar(TS[:, 0], TS[:, ts_te_col], TS[:, ts_te_err_col], fmt='.', capsize=2, lw=2)
        else:
            plt.errorbar(TS[:, 0], TS[:, ts_te_col], TS[:, ts_te_col] * 0.05, fmt='.', capsize=2, lw=2)
        plot(yt, ty)
        plot(ylt, tl, '.')

    del den, nfield, r_obj, ph_max, beam_tx, beam_rx

    """ Save to file"""
    with h5py.File(filename_res + '.h5', 'r+') as hf:
        nfull = len(hf['khi'])

        hf['x0'].resize((nfull + 1,))
        hf['x0'][-1] = x0

        hf['y0'].resize((nfull + 1,))
        hf['y0'][-1] = y0

        hf['khi'].resize((nfull + 1,))
        hf['khi'][-1] = np.sqrt(np.mean((Ph[:, ph_col] - (ph_vac_full - ph_end_full) / 2 / np.pi) ** 2))

        hf['rlcs'].resize((nfull + 1,))
        hf['rlcs'][-1] = rlcs

        for ele_key, ele in zip(('delta', 'trian', 'elong'), (delta, trian, elong)):
            hf[ele_key].resize((nfull + 1, 5))
            hf[ele_key][-1, :len(ele)] = ele

        min_i = np.argmin(hf['khi'])

        x00 = hf['x0'][min_i]
        """        
        y00 = hf['y0'][min_i]
        delta00 = hf['delta'][min_i, 0]
        elong00 = hf['elong'][min_i, 0]+hf['elong'][min_i, 2] * hf['rlcs'][min_i]**2
        trian00 = hf['trian'][min_i, 2]*rlcs**2
        """
        del min_i

        hf.close()

    print("In file stored {:d}".format(nfull + 1) + ' steps')
    print("Total time is {:4.2f} s".format(t[-1] - t[0]))
    # Clean up
    t.resize((0,))
    del ph_end_full, x0, y0, delta, trian, elong, elon0, rlcs, nfull, hf

os.remove(filename_stp)
