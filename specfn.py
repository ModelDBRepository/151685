""" specfn.py - Average time-frequency energy representation using Morlet wavelet method
    This code was modified from the 4D toolbox for MATLAB by Ole Jensen.
"""

import numpy as np
import scipy.signal as sps
import fileio as fio

# MorletSpec class to calculate Morlet Wavelet-based spectrogram
class MorletSpec():
    """ MorletSpec(). Calculates Morlet wavelet spec at 1 Hz central frequency steps
        tsvec: time series vector
        fs: sampling frequency in Hz

        Usage:

        spec = MorletSpec(tsvec, fs)
    """
    def __init__(self, tsvec, fs):
        self.fs = fs

        # this is in s
        self.dt = 1. / fs

        n_ts = len(tsvec)

        # Import dipole data and remove extra dimensions from signal array.
        # in ms
        self.tvec = 1000. * np.arange(0, n_ts, 1) * self.dt + self.dt
        self.tsvec = tsvec

        # maximum frequency of analysis
        # Add 1 to ensure analysis is inclusive of maximum frequency
        self.f_max = 120.

        # cutoff time in ms
        self.tmin = 50.

        # truncate these vectors appropriately based on tmin
        if self.tvec[-1] > self.tmin:
            # must be done in this order! timeseries first!
            tmask = (self.tvec >= self.tmin)
            self.tsvec = self.tsvec[tmask]
            self.tvec = self.tvec[tmask]

            # Array of frequencies over which to sort
            self.f = np.arange(1., self.f_max, 1.)

            # Number of cycles in wavelet (>5 advisable)
            self.width = 7.

            # Generate Spec data
            self.TFR = self.__traces2TFR()

        else:
            print("tstop not greater than %4.2f ms. Skipping wavelet analysis." % self.tmin)

    def __traces2TFR(self):
        self.S_trans = self.tsvec.transpose()

        # preallocation
        B = np.zeros((len(self.f), len(self.S_trans)))

        if self.S_trans.ndim == 1:
            for j in range(0, len(self.f)):
                s = sps.detrend(self.S_trans[:])
                B[j, :] += self.__energyvec(self.f[j], s)

            return B

        else:
            for i in range(0, self.S_trans.shape[0]):
                for j in range(0, len(self.f)):
                    s = sps.detrend(self.S_trans[i,:])
                    B[j,:] += self.__energyvec(self.f[j], s)

    def __energyvec(self, f, s):
        """ Return an array containing the energy as function of time for freq f
            The energy is calculated using Morlet wavelets
            f: frequency
            s: signal
        """
        dt = 1. / self.fs
        sf = f / self.width
        st = 1. / (2. * np.pi * sf)

        t = np.arange(-3.5*st, 3.5*st, dt)
        m = self.__morlet(f, t)
        y = sps.fftconvolve(s, m)
        y = (2. * abs(y) / self.fs)**2
        istart = int(np.ceil(len(m)/2))
        iend = int(len(y)-np.floor(len(m)/2)+1)

        y = y[istart:iend]

        return y

    def __morlet(self, f, t):
        """ Morlet wavelets for frequency f and time t
            Wavelet normalized so total energy is 1
            f: specific frequency
            t: time vector for the wavelet
        """
        sf = f / self.width
        st = 1. / (2. * np.pi * sf)
        A = 1. / (st * np.sqrt(2.*np.pi))

        y = A * np.exp(-t**2. / (2. * st**2.)) * np.exp(1.j * 2. * np.pi * f * t)

        return y

# spectral plotting kernel should be simpler and take just a file name and an axis handle
def pspec_ax(ax_spec, f, TFR, xlim):
    """ Spectral plotting kernel for ONE simulation run
        ax_spec: the axis handle.
        f: frequency (Hz)
        TFR: time-frequency representation (the spec)
        xlim: limits on the x-axis

        returns an axis image object
    """
    extent_xy = [xlim[0], xlim[1], f[-1], f[0]]

    pc = ax_spec.imshow(TFR, extent=extent_xy, aspect='auto', origin='upper')
    [vmin, vmax] = pc.get_clim()

    return pc

if __name__ == '__main__':
    x = fio.pkl_load('data/gammaweak/data.pkl')
    fs = 1. / x['p']['dt']
    spec = MorletSpec(x['dipole_L5'], fs)
