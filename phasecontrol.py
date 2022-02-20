import numpy as np
from readphase import readphase
from data_class import pdata
from phaseapprox import phaseapprox
from pancakecalcul_4 import pancakecalcul_4
from phasecalculate import phasecalculate
from chordlength import chordlength
import matplotlib.pyplot as plt


def phasecontrol(filename, splineparam=None):
    """
    Uses readphase phaseapprox and pancakecalcul
    read phase from file filename approximate it and calculate density pancakes
    splineparam is additional parameter p = 1 - variational spline
    p = 0 - straight line least square approximation
    denspancake - array of records: thick, radius, center;
    x, fx - approximated phase;
    success - indicator: 1 if ok, 0 - otherwise;
    """
    global SMALLVALUE
    global APPROXPOINTSNUMBER # number of points in spline approximation of the phase(phaseapprox.m)
    SMALLVALUE = 1E-4
    global LAMBDA, ELECTRONCHARGE, LIGHTVELOS, ELECTRONMASS, OMEGA, DENSCOEFF
    LAMBDA = 0.22 # 2.15E-1; wavelength of the interferometer
    ELECTRONCHARGE = 4.8027E-10
    LIGHTVELOS = 2.9979E10
    ELECTRONMASS = 9.1083E-28
    OMEGA = 2 * np.pi * LIGHTVELOS / LAMBDA
    DENSCOEFF = (ELECTRONCHARGE) ** 2 / OMEGA / LIGHTVELOS / ELECTRONMASS # *2 * np.pi
    phasedata = pdata()

    #   Unsuccessful values
    denspancake = x = fx = xx = dens = None

    success = 1

    success, phasedata = readphase(filename)
    if success == 0:
        'unsuccessful reading'
        return denspancake,x,fx,xx,dens,success

    if splineparam:     # nargin==2
        x, fx = phaseapprox(phasedata, splineparam)
        if isinstance(filename, str):
            #   print initial phase
            initphasefilename = filename[0:-3]
            initphasefilename = initphasefilename + 'phs'
            printinitphase(success, x, fx, initphasefilename)
        denspancake, success = pancakecalcul_4(x, fx)    # !!! - temporary, really without 2
        if success == 0:
            'unsuccessful fitting'
            return denspancake, x, fx, xx, dens, success
        xx = np.linspace(x[0], x[-1], 200)
        phase, dens, success = phasecalculate(denspancake.radius, denspancake.center, denspancake.thick, xx)

        #   Comments from .m file
        #[radii,shift,thick,success] = denspancake(x,dens);
        #%[phase1,dens1,success]=phasecalculate(radii,shift,thick,x);
    else:
        splineparam = 1
        success = 0
        while splineparam > 0 and success == 0:
            x, fx = phaseapprox(phasedata, splineparam)
            xx = x
            denspancake, success = pancakecalcul_4(x, fx)
            if success == 1:
                print('smoothing spline, spline parameter = {0}'.format(splineparam))
                phase, dens, success = phasecalculate(denspancake.radius, denspancake.center,
                                                      denspancake.thick, x)
            splineparam = splineparam - 0.1
        if success == 0:
            'difficult data'

    #return
    #   print density along equator
    densfilename = filename[0:-3]
    densfilename += 'dns'
    success = printequatordens(success, x, dens, densfilename, denspancake)

    #   print pancakes characteristics
    pancakefilename = filename[0:-3]
    pancakefilename += 'crl'
    #%printpancakes(success,denspancake,pancakefilename);

    #   print radius dependence
    nrfilename = 'nr'
    nrfilename += filename[1:-3]
    nrfilename += 'txt'

    success = printnrfile(success, denspancake, nrfilename)
    plt.show()

    del LAMBDA, ELECTRONCHARGE, LIGHTVELOS, ELECTRONMASS, OMEGA, DENSCOEFF
    return


def printinitphase(success, x, fx, initphasefilename):
    """
    print approximated initial phase to the file filename
    """

    tempx, tempfx = (np.ndarray(1) for _ in range(2))
    if success == 1:
        tempx = x
        tempfx = fx
    try:
        with open(initphasefilename, 'wt') as fid:
            print('{0:s}   {1:s}'.format('r', 'phase'), file=fid)
            #print('{0:-6.4f}   {1:-6.4f}  '.format(tempx, tempfx), file=fid)
            for i in range(len(tempx)):
                print('{0:-6.4f}   {1:-6.4f}  '.format(tempx[i], tempfx[i]), file=fid)
    except OSError:
        print(f"Error: cannot create file {initphasefilename}")
        success = 0
        return


def printequatordens(success, x, dens, densfilename, denspancake):
    """
    print density along equator to the file densfilename
    """
    #  rework for python features if possible
    tmp = np.ndarray((2*len(denspancake), 2))
    if success == 1:
        for indx in range(len(denspancake)):
            tmp[indx, 0] = denspancake.center[indx] - denspancake.radius[indx]
            if indx == 0:
                tmp[indx, 1] = denspancake.thick[indx]
            else:
                tmp[indx, 1] = tmp[indx - 1, 1] + denspancake.thick[indx]
        for i in range(len(denspancake)-1, -1, -1):
            indx = 2 * len(denspancake) - i - 1
            tmp[indx, 0] = denspancake.center[i] + denspancake.radius[i]
            if i == len(denspancake) - 1:
                tmp[indx, 1] = tmp[indx - 1, 1]
            else:
                tmp[indx, 1] = tmp[indx - 1, 1] - denspancake.thick[i + 1]

        try:
            with open(densfilename, 'wt') as fid:
                #print('{0:s}   {1:s}'.format('r', 'phase'), file=fid)
                for i in range(len(denspancake)):
                    print('{0:-6.4f}  {1:-e}'.format(tmp[i, 0], tmp[i, 1]), file=fid)
        except OSError:
            print(f"Error: cannot create file {densfilename}")
            success = 0
            return success
    return success


def printnrfile(success, denspancake, nrfilename):
    """
    calculating function n(r) and print it to file
    """

    if success != 1:
        return success
    pointsnumber = len(denspancake)
    r = np.ndarray(pointsnumber+1)
    c = np.ndarray(pointsnumber+1)
    n = np.ndarray(pointsnumber+1)
    outmatrix = np.ndarray((pointsnumber+2, 3))
    for indpancake in range(pointsnumber+1):    #mb pointsnumber+1
        if indpancake == 0:
            r[indpancake] = denspancake.radius[indpancake] + 10 * SMALLVALUE * max(denspancake.radius)
            c[indpancake] = denspancake.center[indpancake]
        else:
            r[indpancake] = denspancake.radius[indpancake - 1]
            c[indpancake] = denspancake.center[indpancake - 1]
        n[indpancake] = 0
        if indpancake > 0:
            for j in range(indpancake):
                n[indpancake] += denspancake.thick[j]

    try:
        with open(nrfilename, 'wt') as fid:
            print('r       c       n', file=fid)
            for i in range(len(r)):
                print('{0:-6.4f}  {1:-6.4f}  {2:-e}'.format(r[i], c[i], n[i]), file=fid)
    except OSError:
        print(f"Error: cannot create file {nrfilename}")
        success = 0
        return success

    #   checking
    phase = np.zeros(len(r)*2)
    x = np.zeros(len(r)*2)
    dens = np.zeros(len(r)*2)
    for indx in range(len(r)):
        x[indx] = c[indx] - r[indx]
        dens[indx] = n[indx]
        x[2*len(r) - indx - 1] = c[indx] + r[indx]
        dens[2*len(r) - indx - 1] = n[indx]
    for indx in range(len(r)):
        for indphase in range(indx, 2*len(r) - indx - 1):
            currlength = chordlength(r[indx], c[indx], x[indphase])
            if indx == 0:
                phase[indphase] = phase[indphase] + currlength*n[indx]*DENSCOEFF
            else:
                phase[indphase] = phase[indphase] + currlength * (n[indx] - n[indx - 1]) * DENSCOEFF

    tmpx = np.linspace(x[0], x[-1], 100)
    tmpphase = np.zeros_like(tmpx)
    maxd = np.max(dens)
    maxstep = np.argmax(dens)

    for indx in range(len(tmpx)):
        step = 0    #   or 0 or 1?
        if tmpx[indx] < x[maxstep]:
            while tmpx[indx] > x[step + 1]:
                step = step + 1
        else:
            while tmpx[indx] < x[-step - 1]:
                step = step + 1
        if step > len(r):
            step = len(r)
        tmpphase[indx] = 0
        for indpancake in range(step):
            currlength = chordlength(r[indpancake], c[indpancake], tmpx[indx])
            if indpancake == 0:
                tmpphase[indx] = tmpphase[indx] + currlength * (n[indpancake]) * DENSCOEFF
            else:
                tmpphase[indx] = tmpphase[indx] + currlength * (n[indpancake] - n[indpancake - 1]) * DENSCOEFF

    plt.figure(3)
    plt.plot(x, dens, 'k-')
    plt.title('density along equator')
    plt.figure(2)
    plt.plot(x, phase, 'kx-')
    plt.title('phase')
    plt.figure(1)
    plt.plot(tmpx, tmpphase, 'k-')
    plt.show()

    return success
