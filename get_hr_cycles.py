import datetime
import pickle
import pandas as pd
import numpy as np
import pycwt as cwt
import scipy
from utils import *

# open patient HR and seizure data
f = open('used_data.pckl', 'rb')
used_data = pd.read_pickle(f)
f.close()


# placeholder variables
compliant_patients = {}
wavelet_plot = {}
table_info = {}
fft_data = {}
heart_rate_cycles = pd.DataFrame(columns = ['patient', 'f', 'type'])

# wavelet and filter variables                    
mother = cwt.Morlet(6)
order = 2; fs=60

for patient in used_data:

    [data, seizures] = used_data[patient] # seizure variable in control patients is empty dataframe
    df = data.copy()
    
    df['value'] = (df['value'] - df['value'].mean())/df['value'].std() #standardization
    df['value'][pd.isnull(df['value'])] = df['value'].mean() # interpolation
    
    # custom frequencies with maximum scale n (where n is 1/4 the length of the data) - i.e. we need to observe 4 cycles
    n = (df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]).total_seconds()/3600/4
    freqs = np.append(np.arange(2.4, 31.2, 1.2), np.arange(31.2, 48, 2.4))
    freqs = np.append(freqs, np.arange(2.2*24, 4*24 + 4.8, 4.8))
    freqs = np.append(freqs, np.arange(5*24, int(n), 12))
    freqs = (1/freqs)
    
    # wavelet
    y = df.resample('5Min', on='timestamp').mean().reset_index().value.to_numpy()
    dt = 1/12
    alpha, _, _ = cwt.ar1(y) # lag 1 autocorrelation for significance
    wave, scales, freqs, coi, fft, fftfreqs = cwt.cwt(signal = y, dt = dt, wavelet = mother, freqs = freqs)
    power = np.abs(wave) ** 2
    fft_power = np.abs(fft) ** 2
    period = 1 / freqs
    glbl_power = power.mean(axis=1)
    dof = y.size - scales  # Correction for padding at edges
    var = y.std()**2
    glbl_signif, tmp = cwt.significance(var, dt, scales, 1, alpha, significance_level=0.99, dof=dof, wavelet=mother)
    signif, fft_theor = cwt.significance(1.0, dt, scales, 0, alpha, significance_level=0.99, wavelet=mother)


    # Find peaks that are significant
    xpeaks = []; powers = []
    ind_peaks = scipy.signal.find_peaks(var * glbl_power)[0]
    for i in ind_peaks:
        peak = [var * glbl_power > glbl_signif][0][i]
        if peak:
            if period[i] not in xpeaks:
                xpeaks.append(period[i])
                powers.append([var * glbl_power][0][i])

    # keep only stongest peak if there is a peak within +/- 33% of another peak
    xpeaks = np.array(xpeaks)  
    new_xpeaks = {}
    for peak in xpeaks:
        ints2 = np.where(np.logical_and(xpeaks>=peak-0.33*peak, xpeaks<=peak+0.33*peak))
        # is the peak in another peaks BP filter?
        other = [i for i,p in enumerate(xpeaks) if peak >= p - 0.33*p and peak <= p + 0.33*p]
        ints2 = set(np.array(list(ints2[0]) + other))
        if len(ints2):
            # if there is a peak within +/- 33%, check the power of it, choose highest
            max_peak = xpeaks[[var * glbl_power][0].tolist().index(np.max([[var * glbl_power][0][i] for i in ints2]))]
            new_xpeaks[peak] = max_peak
    xpeaks = sorted(set(new_xpeaks.values()))

    # store wavelet data
    wavelet_data = pd.DataFrame()
    wavelet_data['period'] = period
    wavelet_data['power'] = var * glbl_power
    wavelet_data['sig'] = glbl_signif
    wavelet_data['peak'] = np.zeros(len(wavelet_data))
    for peak in xpeaks:
        wavelet_data['peak'][wavelet_data.period == nearest(wavelet_data.period, peak)] = 1
    wavelet_plot[patient] = [wavelet_data, xpeaks]
    f = open('wavelet_plot_data.pckl', 'wb')
    pickle.dump(wavelet_plot, f)
    f.close()
    
    # store fourier transform data
    fft_period, fft, fft_sig = 1./fftfreqs, var * fft_power, var * fft_theor
    fft_data[patient] = [fft_period, fft, fft_sig]
    f = open('fft_data.pckl', 'wb')
    pickle.dump(fft_data, f)
    f.close()
    
    # add heart rate cycles to heart_rate_cycles df
    for f in xpeaks:
        if round(f) == 24:
            peak_type = 'circadian'
        elif f >=(5*24) and f <=(9*24):
            peak_type = 'weekly'
        elif f >=(28*24) and f <=(32*24):
            peak_type = 'monthly'
        else:
            if f<16:
                peak_type = 'ultradian'
            else:
                peak_type = 'multiday'
        heart_rate_cycles = heart_rate_cycles.append({'patient':patient, 'f':f, 'type':peak_type}, ignore_index=True)
    
    
    # if patient has more than 20 seizures, do seizure analysis
    if len(seizures) >= 20:
        filtered = pd.DataFrame()
        filtered['timestamp'] = df['timestamp']
        for f in xpeaks:
            lowcut = 1/(f + (1/3)*f)
            highcut = 1/(f - (1/3)*f)
            # Create a new dataframe to store the filtered data around that frequency, and then corresponding phases and seizures
            filtered_data = butter_bandpass_filter(df['value'], lowcut, highcut, fs, order=order)
            hilb_phases = np.angle(scipy.signal.hilbert(filtered_data)) + np.pi
            analytic_signal = scipy.signal.hilbert(filtered_data)
            amplitude_envelope = np.real(analytic_signal)
            filtered[str(f) + '_value'] = filtered_data
            filtered[str(f) + '_phase'] = hilb_phases
            filtered[str(f) + '_amp'] = amplitude_envelope
        
        filtered = filtered.resample('1H', on='timestamp', label='right').mean().reset_index()
        filtered['seizure'] = np.zeros(len(filtered))
        for sz in seizures.timestamp.to_list():
            time = nearesttime(filtered.timestamp.to_list(), sz)
            filtered['seizure'][filtered.timestamp == time] += 1
        compliant_patients[patient] = filtered
            
        f = open('filtered_phases.pckl', 'wb')
        pickle.dump(compliant_patients, f)
        f.close()

heart_rate_cycles.to_csv('HR_Cycles.csv')