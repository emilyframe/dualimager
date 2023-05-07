# Libraries
import numpy as np
import sys
sys.path.append('/Users/eframe/dmi/src')
from codedAperture import getMaskArray
import time
import tables

# Fixed Parameters
T = 2.4                       # mask thickness (mm)
F = [0, 0, 300]               # focal point, converging (-F-T), diverging (+F) (mm)

# Initializing Output File
outfile = '/Users/eframe/dmi/divergemap300.h5'

# Getting Mask Array, Flip = True for Converging Mask
mFile = '/Users/eframe/dmi/mask.mat'
mP = getMaskArray( maskFile = mFile, padDim = 2, pixelWidth = 2, flip = False )

# Pixels to Evaluate on the Mask Plane at Z = 0 (mm)
dX, dY = np.mgrid[ -69.75:70.25:0.5, -69.75:70.25:0.5 ]
dZ = np.zeros( len( dX.flatten() ) )
dP = np.array( [ dX.flatten(), dY.flatten(), dZ ] ).T

# Getting Phi and Theta from a Cartesian Grid
gX, gY = np.mgrid[ -1:1:0.01, -1:1:0.01 ]
gZ = np.sqrt( 1 - ( gX**2 + gY**2 ) )
grid = np.array( [ gX.flatten(), gY.flatten(), gZ.flatten() ] ).T

# Building the Lookup Table Containing Attenuation Lengths
tme = time.time()
response = np.zeros((len(grid), len(dP)), dtype='float16')
for i in np.arange( len(grid) ):
    print(i)
    R = T / grid[i, 2]
    wx = R * grid[i, 0]
    wy = R * grid[i, 1]
    w = [wx, wy, -T]
    omega = np.linalg.norm(w)

    try:
        num = int(omega / 0.2)
        steps = np.linspace(0, omega, num)
        width = steps[-1] - steps[-2]
        RArray = np.zeros((len(dP), len(steps)))
    except:
        RArray = []

    if len(RArray) > 0:
        for j in np.arange( steps.shape[0] ):
            Rj = steps[j] / grid[i, 2]
            wxj = Rj * grid[i, 0]
            wyj = Rj * grid[i, 1]
            wzj = Rj * grid[i, 2]
            wj = [wxj, wyj, -wzj]

            x = dP + wj
            alpha =  ( - F[2] ) / ( x[:, 2] - F[2] )
            y = alpha[:, None] * ( x - F ) + F

            idxx = np.floor( y[:, 0] / 2 ) * 2
            idyy = np.floor( y[:, 1] / 2 ) * 2

            idx =  ( ( idxx - mP['x'][0, 0] ) / ( mP['x'][1, 0] - mP['x'][0, 0] ) ).astype(int)
            idy =  ( ( idyy - mP['y'][0, 0] ) / ( mP['y'][0, 1] - mP['y'][0, 0] ) ).astype(int)

            mask4 = ( ( idx >= 0 ) & ( idy >= 0 ) ) & \
                    ( ( idx < len( mP['x'][:, 0] ) ) & ( idy < len( mP['x'][:, 0] ) ) )
            mask5 = ~mask4

            val21 = mP[idx[mask4], idy[mask4]]['val'] ^ 1

            RArray[:, j][mask4] = val21
            RArray[:, j][mask5] = 0

        zeros = RArray.shape[1] - np.count_nonzero(RArray, axis=1)
        response[i] = zeros * width

output = tables.open_file(outfile, 'w')
output.create_array('/', 'Matrix', response)
print(time.time() - tme)
