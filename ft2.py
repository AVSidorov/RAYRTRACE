import numpy as np

import den as dn
from beam import Beam, field_result

r_dia = 7.85
r_ant = 10
x_cx = 34.7
y_cx = -9.2


def ants():
    ant = np.array([x for x in np.linspace(-6.9, 6.9, 7)])
    ant = np.vstack((ant, np.sqrt(r_ant ** 2 - ant ** 2)))
    ant = ant.T
    return ant


def limiter(r_dia=r_dia, theta=360):
    if isinstance(theta, (int, np.int64)):
        theta = np.linspace(0, 2 * np.pi, theta, endpoint=False)
    return r_dia * np.cos(theta), r_dia * np.sin(theta)


def phases(den, x, y, *, ant_tx=None, ant_rx=None, a_set=None, nst=-1, nray=251):
    Beam._nray = nray
    if ant_tx is None:
        ant_tx = ants()
        ant_tx[:, 1] *= -1.

    if ant_rx is None:
        ant_rx = ant_tx.copy()
        ant_rx[:, 1] *= -1.

    # lists for beams
    beamsTX = list()
    beamsRX = list()

    if nst > 0:
        a_set = np.linspace(0, 1, nst + 1)

    ph_by_max = np.ndarray((0,))
    ph_by_sig = np.ndarray((0,))

    a_set = np.atleast_1d(a_set)

    for a in a_set:
        for _ in (beamsTX, beamsRX):
            _.clear()

        nfield = dn.Nfield(den * a, x=x, y=y)

        for tx, rx in zip(ant_tx, ant_rx):
            beamsTX.append(Beam(x0=tx[0], y0=tx[1], kx0=0, ky0=1, n_field=nfield))
            beamsRX.append(Beam(x0=rx[0], y0=rx[1], kx0=0, ky0=-1, n_field=None))

        for beam_tx, beam_rx in zip(beamsTX, beamsRX):
            _, _, _, ph_max, signal, *_ = field_result(beam_tx, beam_rx)
            ph_by_max = np.append(ph_by_max, ph_max)
            ph_by_sig = np.append(ph_by_sig, np.angle(signal))

        del nfield, ph_max, signal

    ph_by_max = ph_by_max.reshape((-1, len(beamsTX)))
    ph_by_sig = ph_by_sig.reshape((-1, len(beamsTX)))

    # if 0. in a_set:
    #     ph_by_max = (ph_by_max[a_set == 0.] - ph_by_max)/2/np.pi
    #     ph_by_sig = np.unwrap(ph_by_sig - ph_by_sig[a_set == 0.], axis=0)/2/np.pi

    return ph_by_max, ph_by_sig


def cx_chord(angle, nst=100):
    r_cx = np.sqrt(x_cx ** 2 + y_cx ** 2)
    e0 = -x_cx - 1j * y_cx
    e0 *= 1. / np.abs(e0)
    a = np.linspace(r_cx - r_dia, r_cx + r_dia, nst)
    e = np.exp(1j * (np.angle(e0) - angle * np.pi / 180))
    return x_cx + e.real * a, y_cx + e.imag * a


def ts_chord(nst=100):
    return np.full((nst,), 1.5), np.linspace(-np.sqrt(r_dia ** 2 - 1.5 ** 2), np.sqrt(r_dia ** 2 - 1.5 ** 2), nst)


def chord_min_rho(x, y, r_obj):
    r_max = np.sqrt(x ** 2 + y ** 2).max()
    rho_max = r_max / r_obj.r_lcs * 1.1
    return np.min(r_obj.cartesian2rho((x, y), rho=np.linspace(0, rho_max, x.size)))
