# Imported Libraries
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import math as m

maskType = np.dtype({'names':['x', 'y', 'val'], 'formats':[np.float32, np.float32, np.int]}, align=True)

def getMaskArray(maskFile, padDim, pixelWidth, flip=False):
    """ Builds a mask array with dimensions equivalent to the dimensions of
    the mask input file plus the padDim, which pads the array with zeros. The
    array is centered at (0, 0, 0) with pixel size = pixelWidth (mm). Each pixel
    has a corresponding value (either 1 == closed or 0 == open). If flip is true,
    the mask is converging.
    """
    data = scipy.io.loadmat(maskFile)
    mask = data['Mopt']
    if flip is True:
        mask = np.flipud(mask)
    mask = np.pad(mask, (padDim, padDim), 'constant', constant_values=1)
    mask = np.pad(mask, (1, 1), 'constant') # padding with zero for faster coding purpose
    nRows = mask.shape[0]
    nColumns = mask.shape[1]
    maskArray = np.zeros(( nRows, nColumns ) , dtype = maskType)
    for i in np.arange( nRows ):
        x = i * pixelWidth - (nRows / 2) * pixelWidth
        for j in np.arange( nColumns ):
            y = j * pixelWidth - (nColumns / 2) * pixelWidth
            maskArray[i][j]['x'] = x
            maskArray[i][j]['y'] = y
            maskArray[i][j]['val'] = mask[i][j]
    return maskArray

def computeMLEM(sysMat, counts, sens, eps, nIter=10):
    """this function computes iterations of MLEM it returns the image after nIter iterations

    sysMat is the system matrix, it should have shape: (n_measurements, n_pixels)
    It can be either a 2D numpy array, numpy matrix, or scipy sparse
    matrix

    counts is an array of shape (n_measurements) that contains the number
    of observed counts per detector bin

    sens_j is the sensitivity for each image pixel
    if this is None, uniform sensitivity is assumed
    """

    nPix = sysMat.shape[1]
    lamb = np.ones(nPix)

    iIter = 0
    while iIter < nIter:
        sumKlamb = sysMat.dot(lamb) + 10e-20
        outSum = (sysMat * counts[:, np.newaxis]).T.dot(1 / sumKlamb)
        lamb = (outSum * lamb) * (sens / (sens ** 2 + max(sens) ** 2 * eps ** 2))
        iIter += 1

    return lamb

def make2DMesh(x, y, vals, vmin, vmax):
    """generates 2D mesh plot
    """
    fig, ax = plt.subplots()
    ax.set_xlabel('X (mm)', fontsize=15)
    ax.set_ylabel('Y (mm)', fontsize=15)
    ax.tick_params(labelsize=15)
    # ax.set_xlim(-30, 10)
    # ax.set_ylim(-20, 20)
    if (x is False) and (y is False):
        im = ax.pcolormesh(vals.T, vmin=vmin, vmax=vmax, shading= 'gouraud')
    else:
        im = ax.pcolormesh(x, y, vals.T, vmin=vmin, vmax=vmax, shading= 'gouraud')
    cbar = fig.colorbar(im, format='%.0e')
    cbar.set_label(label='Intensity', rotation=270, fontsize=15, labelpad=25)
    cbar.ax.tick_params(labelsize=15)
    return fig, ax, im

def rebin(a, shape):
    """rebins/compresses a 2D array into the given shape by averaging over
    the pixels
    """
    sh = shape[0],a.shape[0]//shape[0],shape[1],a.shape[1]//shape[1]
    return a.reshape(sh).mean(-1).mean(1)

def rebin2(a, shape):
    """rebins/compresses a 3D array into the given shape by averaging over
    the pixels
    """
    sh = shape[0], shape[1],a.shape[1]//shape[1],shape[2],a.shape[2]//shape[2]
    return a.reshape(sh).mean(-1).mean(2)

def oversample(a, n):
    """oversamples bins in an array by repeating the elements n x n times; lengh
    of the array must have an integer square root.
    """
    a1 = a.repeat( int( n ), axis=0 )
    a2 = a1.reshape( int( np.sqrt( len( a ) ) ), int( n * np.sqrt( len( a ) ) ) )
    a3 = a2.repeat( int( n ), axis=0 )
    return a3
