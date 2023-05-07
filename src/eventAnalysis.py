import numpy as np
import pandas as pd
from itertools import tee

# Detector strip mappings
PITCH = 2 # mm
DETECTOR_SEPARATION = 10 # mm
DETECTOR_THICKNESS = 15 # mm
NUMCH = 38

# Depth interpolation
m = 0.3215503
b = 7.9596899
depth_lin_interp = lambda x: m*x + b
INTERPFNS = [depth_lin_interp]*2

interactionType = np.dtype({'names':['energy', 'x', 'y', 'z', 'dT', 'dt50',\
                                     'dE', 'det', 'z_ref'],\
                            'formats':[np.float32, np.float32, np.float32,\
                                       np.float32, np.float32, np.float32,\
                                       np.float32, np.uint8, np.bool]},\
                                       align=True )

et50_energy_type = np.dtype({"names":['timestamp', 'ADC_value', 'detector', 'trigger',\
                            'rid', 't50', 'energy'], "formats":['<u8', '<f4', '<u2', '<u2',\
                            '<u4', '<f4', '<f4']}, align=True)

def repeated(func, n, x):
    """ Repeats a func(x) n times
    """
    for i in range(n):
        func(x)

def double(iterable):
    """ s -> (s0,s1), (s1,s2), (s2, s3), ...
    """
    a, b = tee(iterable)
    repeated(next, 1, b)
    return list(map(list, zip(a, b)))

def f7(seq):
    """ Similar to the set function, except does not reorder elements
    """
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]

def load_calibration(filename):
    """Loads calibration file with linear calibration values for each detector strip
    """
    df = pd.read_csv(filename)
    p = df.values[:]
    slopes, intercepts = ([] for i in range(2))
    for fit in p:
        slopes.append(fit[1])
        intercepts.append(fit[2])
    return slopes, intercepts

def edata_with_t50_and_energy(edata):
    """Creates t50 and energy fields in the event dataset
    """
    out = np.zeros_like(edata, dtype=et50_energy_type)
    out['timestamp'] = edata['timestamp']
    out['detector'] = edata['detector']
    out['ADC_value'] = edata['ADC_value']
    out['trigger'] = edata['trigger']
    out['rid'] = np.arange(len(edata))
    return out

def edata_threshold(edata, trigger=1, adc=25):
    """Filters edata based on trigger and adc values and sorts edata by timestamp
    """
    edata['rid'] = np.arange(len(edata))
    edata = edata[edata.trigger == trigger]
    edata = edata[edata.ADC_value >= adc]
    edata = edata[edata.detector % 38 != 0]
    edata.sort(order='timestamp')
    return edata

def lump_edata(edata, twindow=30):
    """ Finding lower and upper indices of events that lie within the specified time window
    """
    t = np.array(double(edata['timestamp']))
    tmask = (t[:, -1] - t[:, 0] > twindow)
    tmask = np.reshape(np.repeat(tmask, 2), (-1, 2))

    # Applying Time Mask
    T = np.ma.masked_array(t, tmask).filled(0)

    # Finding min and max indices of events that lie within the time window
    indices = np.sort(list(set(np.argwhere(T == 0)[:, 0])))
    idx1, idx2 = ([] for i in range(2))
    i1 = -1
    for i in indices:
        i2 = i
        if len(T[i1+1:i2]) > 0:
            idx1.append(i1+1)
            idx2.append(i2+1)
        i1 = i

    return idx1, idx2

def inge1(ev):
    """Checks if strip numbers in det array are in detector 1
    """
    dets = ev['detector']
    return ev[(dets < NUMCH) | ((dets >= NUMCH*2)&(dets < NUMCH*3)) ]

def inge2(ev):
    """Checks if strip numbers in det array are in detector 2
    """
    dets = ev['detector']
    return ev[((dets >= 1*NUMCH)&(dets < 2*NUMCH)) | (dets >= NUMCH*3) ]

def onAC(ev):
    """Checks if strip numbers in det array are on the AC side
    """
    return ev[ev.detector >= 2*NUMCH]

def onDC(ev):
    """Checks if strip numbers in det array are on the DC side
    """
    return ev[ev.detector < 2*NUMCH]

def checkForEnergyMatch(en1, en2, sigma=2):
    """Check to see if the two energies agree to within sigma*sqrt of the
     maximum input energy.
     """
    maxEnergy = max((en1, en2))
    other = min((en1, en2))
    if other >= maxEnergy - sigma * np.sqrt(np.abs(maxEnergy)):
        retVal = True
    else:
        retVal = False
    return retVal

def is_single(event):
    """ Determines whether an event is a singles event in the first detector. Allows for charge sharing.
    """
    mask = event['trigger'] == 1
    fired = event[mask]

    d1 = inge1(fired)
    d1_AC = onAC(d1)
    d1_DC = onDC(d1)

    lensAC = len(d1_AC)
    lensDC = len(d1_DC)

    try: check1 = np.abs(int(d1_AC.detector[0]) - int(int(d1_AC.detector[-1]))) <= 1
    except: check1 = False
    try: check2 = np.abs(int(d1_DC.detector[0]) - int(int(d1_DC.detector[-1]))) <= 1
    except: check2 = False
    try: check3 = ( lensAC + lensDC ) == len(fired)
    except: check3 = False

    e1_AC = np.sum( d1_AC['ADC_value'] )
    e1_DC = np.sum( d1_DC['ADC_value'] )

    if ( lensAC >= 1 and lensAC <= 2) and ( lensDC >= 1 and lensDC <= 2) and \
    check1 == True and check2 == True and check3 == True:
        match = checkForEnergyMatch(e1_AC, e1_DC, sigma=2)
        if np.sum(match) == 1:
            return True
        else: return False
    else:
        return False

def is_double(event):
    """ Determines whether an event is a doubles event
    """

    mask = event['trigger'] == 1
    fired = event[mask]

    d1 = inge1(fired)
    d2 = inge2(fired)
    d1_AC = onAC(d1)
    d1_DC = onDC(d1)
    d2_AC = onAC(d2)
    d2_DC = onDC(d2)

    if ( len(d1_AC) == len(d1_DC) ) & ( len(d2_AC) == len(d2_DC) ):
        lens = ( len(d1_AC), len(d2_AC) )
        e1_AC = d1_AC['ADC_value']
        e1_DC = d1_DC['ADC_value']
        e2_AC = d2_AC['ADC_value']
        e2_DC = d2_DC['ADC_value']

        match = np.zeros(2)
        if lens == (2, 0):
            match[0] = checkForEnergyMatch(min(e1_AC), min(e1_DC), sigma=2)
            match[1] = checkForEnergyMatch(max(e1_AC), max(e1_DC), sigma=2)
        elif lens == (0, 2):
            match[0] = checkForEnergyMatch(min(e2_AC), min(e2_DC), sigma=2)
            match[1] = checkForEnergyMatch(max(e2_AC), max(e2_DC), sigma=2)
        elif lens == (1, 1):
            match[0] = checkForEnergyMatch(e1_AC[0], e1_DC[0], sigma=2)
            match[1] = checkForEnergyMatch(e2_AC[0], e2_DC[0], sigma=2)

        if np.sum(match) == 2:
            return True
        else:
            return False

    else:
        return False

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx
