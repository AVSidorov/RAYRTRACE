from data_class import pdata
from scipy.interpolate import CubicSpline
from scipy.misc import derivative
import numpy as np

def endscalculate (phasedata, indexes):
    """
    in case of non zero end calculating zero end by variational spline

    """
    success = 1

    phasespline = CubicSpline(phasedata.r, phasedata.ph, bc_type='natural')
    splinder = phasespline.derivative()
    out = splinder(phasedata.r)
    firstder = out[1]
    lastder = out[2]

    if indexes[0] == len(phasedata.r) or indexes[len(indexes)-1] == len(phasedata.r):
        if lastder >= 0:
            success = 0
            return phasedata, success
        k = lastder
        b = phasedata.ph[-1] - k * phasedata.r[-1]
        r = -b / k
        phasedata.r.append(r)
        phasedata.ph.append(0)
        #   Matlab: phasedata(length([phasedata.r])+1).r=r;
        #           phasedata(length([phasedata.r])).v=0;
    if indexes[0] == 0:
        if firstder <= 0:
            success = 0
            return phasedata, success
        k = firstder
        b = phasedata.ph[0] - k * phasedata.r[0]
        r = -b / k
        data = pdata([r], [0])
        for i in range(len(phasedata.r)+1):
            data.r[i+1].append(phasedata.r[i])
            data.ph[i+1].append(phasedata.ph[i])
        del phasedata
        phasedata = data

    return phasedata, success
