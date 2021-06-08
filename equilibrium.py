import numpy as np
import scipy.interpolate as interp
import pandas


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


def read_asta_tab(filename):
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
