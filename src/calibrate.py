from astropy.convolution import convolve, Gaussian1DKernel, Box1DKernel
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import tables

def gauss(x, h, mu, sig):
    """ Gaussian function
    """
    return h*np.exp(-((x-mu)/sig)**2)

def exponential(t, A, r):
    """ Exponential function
    """
    return A * np.exp(-r * t)

def gaussfilter(counts, std=2, plot=True, xlim=False):
    """Apply a Gaussian filter to spectral data.
    Parameters
    ===========
    counts:         array
                    Spectrum data.
    std:            float
                    Rough estimate of peak standard deviation.
    plot:           boolean
                    If true, plots data with gauss filter
    xlim:           array-like
                    Sets x limit for plot
    Outputs
    ========
    gauss_smooth:   array
                    Smoothed spectrum with Gaussian filter.
    """

    gaussfilter = convolve(counts, Gaussian1DKernel(std))
    if plot is True:
        plt.plot(gaussfilter, 'r', linewidth=3, label='Gaussian Filter')
        plt.plot(counts, 'k', label='Experimental Data')
        plt.xlabel('channel', fontsize=20)
        plt.ylabel('counts', fontsize=20)
        plt.tick_params(labelsize=20)
        plt.legend(fontsize=15)
        if xlim:
            plt.xlim(xlim[0], xlim[1])
        plt.show()
    return gaussfilter

def maxpeak(counts):
    """Find max peak locations in spectral data.
    Parameters
    ===========
    counts:     array
                Spectral data.
    Outputs
    ========
    maxpeak:    array
                Channels corresponding to maximum peaks
    """
    dy = np.diff(counts)
    maxpeak = []
    for i in np.arange(dy.size-1):
        if dy[i]>0 and dy[i+1]<0:
            maxpeak.append([counts[i+1], i+1])
    maxpeak = np.array(sorted(maxpeak, reverse=True))[:,1]
    return maxpeak

def bin_centers(bin_edges):
    """Gets bin centers from bin edges
    Parameters
    ===========
    bin_edges:    array
                  bin edges from histogram
    Outputs
    ========
    bin_centers:  array
                  bin centers
    """
    lower_edges = np.resize(bin_edges, len(bin_edges)-1)
    bin_centers = lower_edges + 0.5*np.diff(bin_edges)
    return bin_centers

def roi(maxpeak, res):
    """Finds ROI based on given peakchannel and energy resoluton
    Parameters
    ===========
    peakchannel:    integer
                    center of peak
    res:            float
                    energy resolution of detector
    Outputs
    ========
    low:            integer, index
                    lower bound of roi
    high:           integer, index
                    upper bound of roi
    """
    sig = int(maxpeak*res)+1
    low = int(maxpeak-sig)
    high = int(maxpeak+sig)
    return low, high


def curvefit(channels, counts):
    """Fits a Gaussian curve to given counts
    Parameters
    ===========
    channels:       array
                    roi channels
    counts:         array
                    roi counts
    Outputs
    ========
    peakchannel:    integer, index
                    lower bound of roi
    high:           integer, index
                    upper bound of roi
    """
    p0 = [int(max(counts)), int(np.mean(channels)), int(np.std(channels))]
    popt, pcov = curve_fit(gauss, channels, counts, p0)
    return popt, pcov


def energycalib(peakchannels, energy, n=1):
    """nth order Polynomial fit
    Parameters
    ===========
    peakchannels:   array-like
                    peakchannels
    energy:         array-like
                    energies corresponding to peakchannels
    n:              integer
                    order of polynomial
    Outputs
    ========
    fit:            array-like
                    nth order polynomial fit
    """
    p = np.polyfit(peakchannels, energy, n)
    fit = np.poly1d(p)
    print('Calibration Model: %s' %fit)
    plt.plot(peakchannels, energy, 'r.', markersize=15, label='calibration data')
    plt.plot(peakchannels, fit(np.array(peakchannels)), 'k-', label = 'linear fit: %s' %fit)
    plt.xlabel('channel', fontsize=20)
    plt.ylabel('energy (keV)', fontsize=20)
    plt.tick_params(labelsize=20)
    plt.legend(fontsize=20)
    plt.show()
    return fit

def plotspectrum(energy, counts, src):
    """Plots gamma spectrum
    Parameters
    ===========
    energy:         array
                    energies
    counts:         array
                    counts
    src:            string
                    radioisotope
    xlim:           float
                    x-axis max limit
    """
    fig, ax = plt.subplots()
    ax.fill(energy, counts, 'k', label=src)
    plt.xlabel('energy (keV)', fontsize=20)
    plt.ylabel('counts', fontsize=20)
    plt.tick_params(labelsize=20)
    plt.ylim(0)
    plt.legend(fontsize=15)
    plt.show()

def readCalibData(file1, file2):
    """reads calibration data where file1 and file2 are data files from the same
    source, e.g. am1.h5 and am2.h5. The output is an 2D array of adc values sorted
    by detector number.
    """
    d1 = tables.open_file(file1, 'r')
    d2 = tables.open_file(file2, 'r')
    events1 = d1.root.EventData.read()
    events2 = d2.root.EventData.read()
    events = np.concatenate((events1, events2))
    adc = events['ADC_value']
    det = events['detector']
    values = []
    for i in range(0, 152, 1):
        mask = (det == i)
        values.append(adc[mask])
    return values

def subBack(events, E, sig, side='both', type='singles'):
    """Subtracts compton background from energy peak
    Parameters
    ===========
    events:         single interaction events
    E:              energy of peak
    sig:            energy resolution of detector at energy peak

    Outputs
    ========
    cnt:            counts in energy peak minus background

    """
    if type == 'singles':
        inside = ( events['energy'] >= E-sig ) & ( events['energy'] < E+sig )
        left = ( events['energy'] >= E-2*sig ) & ( events['energy'] < E-sig )
        right = ( events['energy'] >= E+sig ) & ( events['energy'] < E+2*sig )
    elif type== 'doubles':
        inside = ( events['energy'].sum(axis=1) >= E-sig ) & ( events['energy'].sum(axis=1) < E+sig )
        left = ( events['energy'].sum(axis=1) >= E-2*sig ) & ( events['energy'].sum(axis=1) < E-sig )
        right = ( events['energy'].sum(axis=1) >= E+sig ) & ( events['energy'].sum(axis=1) < E+2*sig )

    if side == 'both':
        cnt = len(events[inside]) - len(events[left]) - len(events[right])
    elif side == 'left':
        cnt = len(events[inside]) - 2 * len(events[left])
    elif side == 'right':
        cnt = len(events[inside]) - 2 * len(events[right])
    elif side == 'none':
        cnt = len(events[inside])
    if cnt < 0: cnt = 0

    return cnt
