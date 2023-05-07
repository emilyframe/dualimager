# Importing Libraries
import tables
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
sys.path.append('/Users/eframe/dmi/src')
import eventAnalysis as ea
from scipy.special import erf

res = np.array([11.3, 6.41, 6.18]) / 2.36 # resolution values for cci-2 doubles events
fit = np.poly1d( np.polyfit([122, 356, 662], res, 1) )
m = 0.3215503
b = 7.9596899
PITCH = 2                       # separation between strips (mm)
DETECTOR_SEPARATION = 10        # separation between Ge1 and Ge2 (mm)
DETECTOR_THICKNESS = 15         # thickness of Ge1 and Ge2 (mm)
NUMCH = 38
geXS = pd.read_hdf('/Users/eframe/dmi/ge-cross-sections.h5', 'vals') # germanimum cross sections

def refine_z(interaction_list):
    """Use t50 to calculate a refined depth position for detector 2.
    Return refined z-values
    """
    new_time_diff = interaction_list['dT'] + interaction_list['dt50']
    new_z = m * new_time_diff + b
    new_z[interaction_list['det'] == 2] += 25
    return new_z


def correct_depth(interaction_list):
    """For every "z" in the interaction list, if it is outside the physical
    boundaries of the detector, for it back to the edge of the detector.
    """
    # Make sure the interactions are not shaped
    orig_shape = interaction_list.shape
    interaction_list = interaction_list.ravel()
    out_list = np.copy(interaction_list)

    # Correct z
    out_list['z'][out_list['z'] < 0.0] = 0.0
    out_list['z'][(out_list['det'] == 1) & (out_list['z'] >= DETECTOR_THICKNESS)] = DETECTOR_THICKNESS
    out_list['z'][(out_list['det'] == 2) & (out_list['z'] <= DETECTOR_THICKNESS + DETECTOR_SEPARATION)] = DETECTOR_THICKNESS + DETECTOR_SEPARATION
    out_list['z'][(out_list['z'] >= 2 * DETECTOR_THICKNESS + DETECTOR_SEPARATION)] = 2 * DETECTOR_THICKNESS + DETECTOR_SEPARATION

    # Cast back to original shape
    out_list = out_list.reshape(orig_shape)
    return out_list

def convertToConeData(events, Es, leverThres):
    """Gets cone data from CCI-2 doubles events
    """
    # Get mu
    Emax = Es - Es / ( 1 + 2 * Es / 511 )
    E = events['energy']
    E2 = E.min(axis=1)
    mask =  np.sort(E)[:,1] > Emax
    E2[mask] = E[mask].max(axis=1)
    coneMu = 1 + 511 * ( 1 / Es - 1 / E2 )

    eMask = ( E2 == E.T ).T

    # Get Interaction Position Order
    pos1 = np.squeeze( np.array( [ events['x'][~eMask], events['y'][~eMask], events['z'][~eMask] ] ) ).T
    pos2 = np.squeeze( np.array( [ events['x'][eMask], events['y'][eMask], events['z'][eMask] ] ) ).T

    # Get Cone Direction
    coneAxes = pos2 - pos1

    norms = np.sqrt( ( coneAxes ** 2 ).sum( axis = 1 ) )
    coneDirs = coneAxes / norms[ :, np.newaxis ]

    # Lever Arm Threshold
    lmask = norms >= leverThres

    return pos1[lmask], pos2[lmask], coneDirs[lmask], coneMu[lmask]

def convertToConeData_SEQ(energy, pos, Es, leverThres):
    """Gets cone data from CCI-2 doubles events
    """

    E1 = energy[:, 0]
    E2 = energy[:, 1]

    coneMu = 1 + 511 * ( 1 / Es - 1 / E2 )

    # Get Interaction Position Order
    pos1 = pos[:,0]
    pos2 = pos[:,1]

    # Get Cone Direction
    coneAxes = pos2 - pos1

    norms = np.sqrt( ( coneAxes ** 2 ).sum( axis = 1 ) )
    coneDirs = coneAxes / norms[ :, np.newaxis ]

    # Lever Arm Threshold
    lmask = norms >= leverThres

    return pos1[lmask], pos2[lmask], coneDirs[lmask], coneMu[lmask]

def coneVoxel(voxelCoord, interPos, coneDir, coneMu, dTheta, binSize):
    L2 = binSize ** 2
    dth2 = dTheta ** 2
    nCones = coneMu.shape[0]
    nPix = voxelCoord.shape[0]
    backproj = np.zeros( ( nCones, nPix ) )

    for i in range( nCones ):
        coneVertexToVoxelDir = interPos[i] - voxelCoord
        R2 = np.sum( coneVertexToVoxelDir ** 2, axis=1 )
        vtsqR2 = np.array(coneDir[i]).dot( coneVertexToVoxelDir.T )
        v = vtsqR2 / np.sqrt( R2 )
        omv2 = 1 - v ** 2
        omv2[omv2 < 0] = 0

        t2 = np.exp( - ( 1 - coneMu[i] * v - np.sqrt( omv2 * ( 1 - coneMu[i] **2 ) ) ) / (dth2 + ( L2 / ( 12 * R2 ) ) ) )
        t1 = np.sqrt( ( L2 + 24 * np.pi * R2 * omv2 ) * ( L2 + 12 * R2 * dth2 ) )

        weightConeToVoxel = (t2 / t1 ) # with sensitivity
#         weightConeToVoxel = R2 ** 0.25 * (t2 / t1 ) # without sensitivity

        backproj[i, :] = weightConeToVoxel

    return backproj

def coneVoxel2(xyz, coneX, coneDir, mu, sig_theta, sig_voxel, R_cut=0.1):
    nCones = mu.shape[ 0 ]
    nPix = xyz.shape[ 0 ]
    backproj = np.zeros( ( nCones, nPix ) )
    Dc = ( 0.014 * 8 ) ** 2 / ( 4 * np.pi ) ** 2
    
    for i in range( nCones ):
        R_vec = coneX[i] - xyz
        R = np.sqrt( np.sum( R_vec**2, axis=1) )
        R_vec = R_vec / R[:, np.newaxis]

        nu = coneDir[i].dot(R_vec.T)
        sin_nu = np.sqrt(1 - nu**2)
        sin_mu = np.sqrt(1 - mu[i]**2)
        kap = nu * mu[i] + sin_mu * sin_nu

        Sigma_meas = sig_theta * sin_mu
        V_i = (np.sqrt(2 * np.pi) * sig_voxel)**3

        mu_3 = (sig_voxel**2 * (1 - nu**2) * mu[i] + R**2 * Sigma_meas**2 * nu) / \
               (sig_voxel**2 * (1 - nu**2) + R**2 + Sigma_meas**2)
        Sig_3 = Sigma_meas**2 * sig_voxel**2 * (1 - nu**2) /\
            (sig_voxel**2 * (1 - nu**2) + R**2 * Sigma_meas**2)

        bp = Dc**2 * sin_nu / np.sqrt(2 * np.pi * R**2 * np.abs(kap) * sin_mu * sin_nu + sig_voxel**2) * \
            V_i / np.sqrt(R**2 * Sigma_meas**2 + sig_voxel**2 * (1 - nu**2)) *\
            .5 * (1 + erf(R * kap / np.sqrt(2) / sig_voxel)) *\
            np.exp(-.5 * R**2 * (mu[i] - nu)**2 / (R**2 * Sigma_meas**2 + sig_voxel**2 * (1 - nu**2))) *\
            (erf((1 + mu_3) / np.sqrt(2) / Sig_3) + erf((1 - mu_3) / np.sqrt(2) / Sig_3)) / \
            (erf((1 + mu[i]) / np.sqrt(2) / Sigma_meas) + erf((1 - mu[i]) / np.sqrt(2) / Sigma_meas))

        bp[R < R_cut] = 0
        backproj[i,:] = bp
    return backproj


def point2Cone(events, Es, P):
    # Get mu
    me = 511.
    Emax = Es - Es / ( 1 + 2 * Es / 511 )
    E = events['energy']
    E2 = E.min(axis=1)
    mask =  np.sort(E)[:,1] > Emax
    E2[mask] = E[mask].max(axis=1)
    coneMu = 1 + me * ( 1 / Es - 1 / E2 )

    muMask = ( coneMu >= -1 ) & ( coneMu <= 1 )
    eMask = ( E2 == E.T ).T

    # Get Interaction Position Order
    pos1 = np.squeeze( np.array( [ events['x'][~eMask], events['y'][~eMask], events['z'][~eMask] ] ) ).T
    pos2 = np.squeeze( np.array( [ events['x'][eMask], events['y'][eMask], events['z'][eMask] ] ) ).T

    # Get Cone Direction
    coneAxes = pos2 - pos1
    vecP = P - pos1

    norms = np.sqrt( ( coneAxes ** 2 ).sum( axis = 1 ) )
    normsP = np.sqrt( ( vecP ** 2 ).sum( axis = 1 ) )

    coneDirs = coneAxes / norms[ :, np.newaxis ]
    pDirs = vecP / normsP[ :, np.newaxis ]

    alpha = np.arccos(np.sum(coneDirs[muMask] * pDirs[muMask], axis=1))
    beta = np.abs(alpha - np.arccos(coneMu[muMask]))
    dmin = normsP[muMask] * np.sin(beta)

    return dmin

def point2ConeInterp(events, Es, P):
    # Get mu
    me = 511.
    Emax = Es - Es / ( 1 + 2 * Es / 511 )
    E = np.array([events['E1'], events['E2']]).T
    E2 = E.min(axis=1)
    mask =  np.sort(E)[:,1] > Emax
    E2[mask] = E[mask].max(axis=1)
    coneMu = 1 + me * ( 1 / Es - 1 / E2 )

    muMask = ( coneMu >= -1 ) & ( coneMu <= 1 )
    eMask = ( E2 == E.T ).T

    eventsx = np.array([events['x1'], events['x2']]).T
    eventsy = np.array([events['y1'], events['y2']]).T
    eventsz = np.array([events['z1'], events['z2']]).T

    # Get Interaction Position Order
    pos1 = np.squeeze( np.array( [ eventsx[~eMask], eventsy[~eMask], eventsz[~eMask] ] ) ).T
    pos2 = np.squeeze( np.array( [ eventsx[eMask], eventsy[eMask], eventsz[eMask] ] ) ).T

    # Get Cone Direction
    coneAxes = pos2 - pos1
    vecP = P - pos1

    norms = np.sqrt( ( coneAxes ** 2 ).sum( axis = 1 ) )
    normsP = np.sqrt( ( vecP ** 2 ).sum( axis = 1 ) )

    coneDirs = coneAxes / norms[ :, np.newaxis ]
    pDirs = vecP / normsP[ :, np.newaxis ]

    alpha = np.arccos(np.sum(coneDirs[muMask] * pDirs[muMask], axis=1))
    beta = np.abs(alpha - np.arccos(coneMu[muMask]))
    dmin = normsP[muMask] * np.sin(beta)

    return dmin

def computeMLEM( sysMat, nIter, sens, eps ):
    nPix = sysMat.shape[1]
    lamb = np.ones(nPix)

    iIter = 0

    if sens is not False:
        while iIter < nIter:
            print(iIter)
            sumKlamb = sysMat.dot(lamb)
            outSum = sysMat.T.dot(1 / sumKlamb)
            lamb = outSum * lamb * ( sens / ( sens ** 2 + max(sens) ** 2 * eps ** 2 ) )
            iIter += 1

    else:
        while iIter < nIter:
            print(iIter)
            sumKlamb = sysMat.dot(lamb)
            outSum = sysMat.T.dot(1 / sumKlamb)
            lamb = outSum * lamb
            iIter += 1

    return lamb

def computeMLEM_TV( sysMat, nIter, nIter2, eps ):
    nPix = sysMat.shape[1]
    lamb = np.ones(nPix)
    iIter, iTter2 = 0, 0
    while iIter < nIter:
        print(iIter)
        sumKlamb = sysMat.dot(lamb)
        outSum = sysMat.T.dot(1 / sumKlamb)
        lamb = outSum * lamb * ( sens / ( sens ** 2 + max(sens) ** 2 * eps ** 2 ) )
        iIter += 1
        while iIter2 < nIter2:
            lamb = np.pad(lamb.reshape(sourceX.shape), 1)
            lambNew = np.zeros_like(lamb)
            for i in np.arange(1, lamb.shape[0] - 1 ):
                for j in np.arange(1, lamb.shape[1] - 1 ):
                    for k in np.arange(1, lamb.shape[2] - 1):
                        sumN = ( lamb[i-1][j][k] + lamb[i+1][j][k] + \
                                    lamb[i][j-1][k] + lamb[i][j+1][k] + \
                                    lamb[i][j][k-1] + lamb[i][j][k+1] ) / 6
                        lambNew[i][j][k] = lamb[i][j][k] - 0.5 * sumN
            lamb = lambNew[1:-1,1:-1,1:-1].flatten()
            iIter2 += 1

    return lamb

# Probability Definitions For Gamma-ray Sequencing

def probCS( E ):
    """Returns the compton scattering cross section in units of barns given incident
    energy E.
    """
    emass = 510.9989461 # electron mass (keV)
    Re = 2.81794 * 10 ** -15 # classical electron radius in meters
    AM = 2 * np.pi * Re ** 2
    ABarn = AM * 10 ** 28
    mina = 2 ** -14
    alp = 2 * E / emass
    alp = max( alp, mina )
    alp1 = 1 + alp
    alp2 = 2 + alp

    term1 = 0.5 * alp2 / ( alp1 * alp1 )
    term2 = np.log( alp1 ) / alp
    coef3 =  alp - ( 1 + 0.5 * alp ) * np.log( alp1 )
    term3 = 8 * coef3 / ( alp * alp * alp )
    terms = term1 + term2 + term3
    cs = ABarn * terms
    return cs

def probKN( Es, E2 ):
    """Returns kleina nishina weighting function given the incident (Es) and
    scattering (E2) photon energy.
    """
    mu = 1 + 511 * ( 1 / Es - 1 / E2 )

    Elow = 0.016
    emass = 510.9989461
    re = 2.81794 * 10 ** -15
    ABarn = 0.5 * re ** 2 * 10 ** 28

    Einc = max( Es, Elow )
    mu1 = min( mu, 1 )
    mu1 = max( mu1, -1 )

    e2 = Einc * emass / ( Einc * ( 1 - mu1 ) + emass )
    Pm = e2 / Einc
    Qm = Einc / e2
    KNmesh = ( Pm * Pm ) * ( Pm + Qm - 1 + mu1 * mu1)

    pKN = ABarn * KNmesh / probCS( Es )
    
    return pKN 

def probPE( E ):
    """Returns photoelectric probability given incident energy E
    """
    i = ea.find_nearest( geXS['energy'].values, E )
    pPE = geXS['pe'][i] / geXS['total'][i]

    return pPE

def probAtten( L, E ):
    """Returns the attenuation probability given the length traveled L (cm) and incident energy E
    """
    i = ea.find_nearest( geXS['energy'].values, E )
    mu_tot = geXS['total'][i] # total interaction probability in cm-1
    pAtten = np.exp( -np.array( mu_tot ) * L  )

    return pAtten

def sequence_probability(energy, pos, Es):
    """Gets probability of interaction sequence from CCI-2 doubles events.
    Es is the assumed incident energy.
    """
    E1 = energy[:,0]
    E2 = energy[:,1]

    # Calculate Photoelectric Absorption Probability
    probPE1, probPE2 = [], []
    for i in np.arange( len( E2 ) ):
        probPE1.append( probPE( E2[i] ) )
        probPE2.append( probPE( E1[i] ) )

    # Calculate Attenuation Probability
    # Finding length travelled in germanium
    pos1, pos2 = pos[ :, 0 ], pos[ :, 1 ]

    L = np.sqrt( ( ( pos1 - pos2 ) ** 2 ).sum( axis = 1 ) ) 

    # Finding events that take place in two detectors
    mask =  ( ( np.abs( pos1[ :,2 ] ) < 15 ) & ( np.abs( pos2[ :, 2 ] ) > 25 ) ) | \
            ( ( np.abs( pos1[ :, 2 ]) > 25 ) & ( np.abs( pos2[ :, 2 ] ) < 15 ) )

    z1, z2 = np.abs( pos1[ :, 2 ] ), np.abs( pos2[ :, 2 ] )

    z1[ z1 <= DETECTOR_THICKNESS ] = DETECTOR_THICKNESS - z1[ z1 <= DETECTOR_THICKNESS ]
    z1[ z1 > DETECTOR_THICKNESS ] = z1[ z1 > DETECTOR_THICKNESS ] - ( DETECTOR_THICKNESS + DETECTOR_SEPARATION )
    z2[ z2 <= DETECTOR_THICKNESS ] = DETECTOR_THICKNESS - z2[ z2 <= DETECTOR_THICKNESS ]
    z2[ z2 > DETECTOR_THICKNESS ] = z2[ z2 > DETECTOR_THICKNESS ] - ( DETECTOR_THICKNESS + DETECTOR_SEPARATION )

    z = np.abs( pos1[ :, 2 ] - pos2[ :, 2 ] )
    theta = np.arccos( z / L )
    l = z1 / ( z / L ) + z2 / ( z / L ) # length traveled in germanium minus air

    L[mask] = l[mask] / 10 # cm
    L[~mask] = L[~mask] / 10 

    probAtt1, probAtt2 = [], []
    for i in np.arange( len(E2) ):
        probAtt1.append( probAtten( L[i], E2[i] ) )
        probAtt2.append( probAtten( L[i], E1[i] ) )

    # Calculating the Klein Nishina Weighting Factor
    probKN1, probKN2 = [], []
    for i in np.arange( len(E2) ):
        probKN1.append( probKN( Es, E2[i] ) )
        probKN2.append( probKN( Es, E1[i] ) )

    P1 = np.array( probAtt1 ) * np.array( probPE1 ) * np.array( probKN1 )
    P2 = np.array( probAtt2 ) * np.array( probPE2 ) * np.array( probKN2 ) 

    return P1, P2
