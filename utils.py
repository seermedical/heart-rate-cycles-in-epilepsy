from scipy.signal import butter, sosfiltfilt
from datetime import timedelta
import pandas as pd
import numpy as np
from astropy import stats as astat
from pycircstat import tests


def drop(data, start, end):
    # Input: dataframe of data, start and end times in pd.Timestamp type with no timezones in the timestamps
    # Output: dataframe of data between with times from [start, end]
    if data is not None and len(data)>0:
        if type(start) != type(data['timestamp'].iloc[0]) and data['timestamp'].iloc[0].tzinfo is None: start = pd.Timestamp(start.replace(tzinfo=None)); end = pd.Timestamp(end.replace(tzinfo=None))
        elif type(start) != type(data['timestamp'].iloc[0]) and data['timestamp'].iloc[0].tzinfo is not None:
            if start.tzinfo is None: print('timezones are incompatible for comparison'); return None
            else: start = pd.Timestamp(start); end = pd.Timestamp(end)
        elif start.tzinfo is not None: start = pd.Timestamp(start.replace(tzinfo=None)); end = pd.Timestamp(end.replace(tzinfo=None))
        check_data = data[data.timestamp >= start]
        return check_data[check_data.timestamp <= end].reset_index(drop = True)
    else:
        return None


# Gets the nearest value in a list of items to the pivot point. Retuns the closest value.
def nearest(items, pivot):
    # Inputs: list of items and pivot point.
    # Ouputs: closest value in list to pivot point.
    return min(items, key=lambda x: abs(x - pivot))


# Function finds closest time to time in list of timestamps
def nearesttime(timestamps, time):
    return min(timestamps, key=lambda x: abs((x - time).total_seconds()))


# This function also gets the nearest time value in a list of timestamps or datetimes.
# It is faster than using the nearest function because it first removes a lot of the unnecessary timestamps using inbuilt pandas functions (within +- 150 seconds of the time (default)).
def nearestvalue(data, time, sec = 150, value_col = 'value'):
    # Input: the dataframe of data with columns ['timestamps', 'value'], the time we want to pivot around and the range of times considered acceptably 'close' (+- 150 seconds is default).
    # If there are no timestamps within +- 150 seconds of the given time value, the function will return None.
    # Output: closest date and closest corresponding value to the time given.
    # Drop off unncessary data
    start_count = time - timedelta(seconds = sec); end_count = time + timedelta(seconds = sec)
    start_time = '{}:{}:{}'.format(start_count.hour, start_count.minute, start_count.second); end_time = '{}:{}:{}'.format(end_count.hour, end_count.minute, end_count.second)
    newdata = data.set_index('timestamp').between_time(start_time, end_time); newdata['timestamp'] = newdata.index; newdata = drop(newdata, start_count, end_count)
    if newdata is None or newdata.empty or len(newdata) == 0:
        return None, None
    else:
        date = nearesttime(newdata['timestamp'], time)
        value = newdata[value_col][newdata[newdata['timestamp'] == date].index.tolist()[0]]
        return date, value
    
    
# Filter functions
# To use: butter_bandpass_filter(signal, lowcut, highcut, fs, order=1)
def butter_bandpass(lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        sos = butter(order, [low, high], analog=False, btype='band', output='sos')
        return sos


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
        sos = butter_bandpass(lowcut, highcut, fs, order=order)
        y = sosfiltfilt(sos, data)
        return y


def lead_sz(filtered, i):
    if i >= 6:
        if 1 in filtered.seizure.iloc[i-6:i].to_list():
            lead_ind = np.min([j for j in range(i-6,i) if filtered.seizure.iloc[j] == 1])
            return [False, lead_ind]
        else:
            return [True, 0]
    else:
        return [True, 0]


def circ_add(a,b):
    if a + b > np.pi*2:
        return a + b - np.pi*2
    else:
        return a + b


# Rayleigh test for uniformity around a circle
def rayleigh(phases):
    return astat.circstats.rayleightest(phases)


# Other circular unifomity tests to compare with rayleigh test
def circ_uniformity(phases, N=1000, test = 'omni'):
    # Inputs: the phases of the dataset
    # The number of times we simulate a random distribution
    p_omni, m = tests.omnibus(phases)
    
    if test == 'omni':
        return p_omni
    else:
        p_rao, tstat_rao, criticalval_rao = tests.raospacing(phases)

        # Do N samples of the next test:
        pvals = []
        for i in range(N):
            uniform_dist = np.random.uniform(size=len(phases))*2*np.pi
            p_kuiper, tstat_kuiper = tests.kuiper(phases, uniform_dist)
            pvals.append(p_kuiper)
        p_kuiper = np.mean(pvals)
        return p_omni, p_rao, p_kuiper


def round_sig(x, sig=2):
    return round(x, sig-int(np.floor(np.log10(abs(x))))-1)
