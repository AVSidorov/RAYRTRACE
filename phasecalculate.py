import numpy as np
from chordlength import chordlength

def phasecalculate(radii, shift, thick, xin):
    """
    Return: phase, dens, success
    calculating phase shift by density pancakes
    checking
    """
    #   Unsuccessful values
    phase = None
    dens = None

    smallpart = 1E-10
    LAMBDA = 0.22  # 2.15E-1; wavelength of the interferometer
    ELECTRONCHARGE = 4.8027E-10
    LIGHTVELOS = 2.9979E10
    ELECTRONMASS = 9.1083E-28
    OMEGA = 2 * np.pi * LIGHTVELOS / LAMBDA
    DENSCOEFF = (ELECTRONCHARGE) ** 2 / OMEGA / LIGHTVELOS / ELECTRONMASS
    #DENSCOEFF = 1
    success = 1

    if len(radii) != len(shift) or len(radii) != len(thick) or xin == []:
        'incorrect data'
        success = 0
        return phase, dens, success

    phase = np.zeros_like(xin)
    dens = np.zeros_like(xin)
    dx = xin[1] - xin[0]

    for indx in range(len(xin)):
        for indpnc in range(len(radii)):
            if shift[indpnc] - radii[indpnc] < xin[indx] - dx * smallpart and shift[indpnc] + \
                    radii[indpnc] > xin[indx] + dx * smallpart:
                #   current chord crosses current pancake
                currlength = chordlength(radii[indpnc], shift[indpnc], xin[indx])
                phase[indx] += currlength * thick[indpnc] * DENSCOEFF
                dens[indx] += thick[indpnc]
    return phase, dens, success

