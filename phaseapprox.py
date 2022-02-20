import numpy as np
#import csaps
from scipy.interpolate import UnivariateSpline
from endscalculate import endscalculate
from data_class import pdata
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot


def phaseapprox(phasedata, splineparam):
    """
    approximate phase data by smoothing spline with parameter splinrparam
    """

    global APPROXPOINTSNUMBER
    APPROXPOINTSNUMBER = 100    # 100; number of approximation points
    #APPROXPOINTSNUMBER=len(phasedata)

    #   approximation
    x = np.linspace(phasedata.r[0], phasedata.r[-1], APPROXPOINTSNUMBER)

    #   exact spline
    #phasespline=csape([phasedata.r], [phasedata.v], 'variational');

    #   smoothing spline
    #functionsplineparam=0.7;%0.15;
    precision = np.ones_like(phasedata.r)
    precision[0] = 1E5
    precision[-1] = 1E5
    #phasespline = csaps.UnivariateCubicSmoothingSpline(phasedata.r, phasedata.ph, smooth=splineparam)
    phasespline = UnivariateSpline(phasedata.r, phasedata.ph, w=precision, s=splineparam)

    fx = phasespline(x)

    if len(x) == len(phasedata.r):
        x = phasedata.r
        fx = phasedata.ph

    #   plotting
    plt.figure(1)
    plt.title('Initial phase and spline approximation')
    #plot([phasedata.r], [phasedata.v], 'ro');
    #plot(x,fx,'bx-');
    plot(phasedata.r, phasedata.ph, 'ko')
    plot(x, fx, 'kx-')
    plt.show()

    return x, fx
