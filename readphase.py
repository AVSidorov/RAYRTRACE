import numpy as np
from endscalculate import endscalculate
from data_class import pdata

def readphase(filename):
    """
    Read file with phase data. calculate end points if they are non-zero
    file - two columns : first column - r, second column - v
    without title string
    """

    smallvalue = 1E-5
    tmp = {'r': [], 'v': []}
    success = 1
    #phasedata = {'r': [], 'ph': []}
    r, ph = (list() for _ in range(2))

    if isinstance(filename, str):
        try:
            with open(filename, "r") as fid:
                for one_line in fid.readlines():
                    one_line = one_line.split()
                    r.append(one_line[0])
                    ph.append(one_line[1])
        except OSError:
            print("Error: there is no such file")
            success = 0
            return success, None
    else:
        pass
        #   # ismatrix(filename)

    r, ph = (np.array(_) for _ in [r, ph])

    #   sort data by radii
    r = r.astype(float)
    ph = ph.astype(float)
    r_arg = np.argsort(r)
    r = np.sort(r)
    for i, arg in zip(range(len(ph)), r_arg):
        ph[i] = ph[arg]

    phasedata = pdata(r, ph)
    del ph, r,  r_arg

    #   checking zeros at the ends and correct it if it's needed
    indexes = list()
    if abs(phasedata.ph[0]) > smallvalue * np.max(phasedata.ph):
        indexes.append(0)
    if abs(phasedata.ph[-1]) > smallvalue * np.max(phasedata.ph):
        indexes.append(len(ph))
    if len(indexes) > 0:
        #   phasedata, success = endscalculate(phasedata, indexes)
        success = phasedata.endscalculate(indexes)
        print('nonzero ends', indexes)

    return success, phasedata
