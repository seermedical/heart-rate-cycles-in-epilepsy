import pickle
import random
import scipy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import timedelta
import pickle
from utils import nearestvalue, drop, lead_sz, circ_add, circ_uniformity, rayleigh, round_sig


# PLOT CONSTANTS
graph_size = (4,1.8)
hist_size = (1.6,1.6)
font_size_tick = 8
marker_sz = 3.5
hr_buffer = 0.1
ax_lw = 0.25
plot_lw = 0.5


# Gather data
f = open('filtered_phases.pckl', 'rb')
filtered_data = pickle.load(f)
f.close()

f = open('used_data.pckl', 'rb')
used_data = pickle.load(f)
f.close()

# placeholder dataframes
sig_phase_locking = pd.DataFrame(columns=['patient', 'f', 'type', 'p_omni', 'p_ray', 'si_val', 'circ_mean', 'multi_comparisons', 'p_random_dist'])
bonferroni_sig_cycles = pd.DataFrame(columns = ['patient', 'f', 'type', 'multi_comparisons', 'p_omni', 'p_random_dist', 'si_val', 'circ_mean'])

'''START PLOTTING'''
# for patient in all_patient_data.keys():
for patient in filtered_data.keys():
        
    
    [data, seizures] = used_data[patient]
    slow = data.resample('1H', label = 'right', on = 'timestamp').mean().reset_index()
    monthly = slow.copy()

    # Apply moving average filters
    data['MA'] = data['value'].rolling(60, min_periods = 5).mean()
    slow['value'] = slow['value'].rolling(48, min_periods = 16).mean()
    monthly['value'] = monthly['value'].rolling(7*24, min_periods = 5*24).mean()


    # Add seizures to fast, slow and monthly dataframes
    data['seizure'] = np.zeros(len(data))
    slow['seizure'] = np.zeros(len(slow))
    monthly['seizure'] = np.zeros(len(monthly))
    for i in seizures['timestamp']:
        date, _ = nearestvalue(data, i)
        data['seizure'][data.timestamp == date] = 1
        date, _ = nearestvalue(slow, i, sec=60*60)
        slow['seizure'][slow.timestamp == date] = 1
        monthly['seizure'][monthly.timestamp == date] = 1

    # find first and last time of fornightly fast plot
    first_time = (data['timestamp'].iloc[-1] - data['timestamp'].iloc[0])/2 + data['timestamp'].iloc[0]
    last_time = first_time + timedelta(days=14)
    while drop(data, first_time, last_time).isnull().value.any():
        first_time += timedelta(days=1)
        last_time += timedelta(days=1)

    # find seizure x (time) and y (hr) values for fast, slow and monthly plots
    slow_sz_x = [slow['timestamp'].iloc[i] for i in slow[slow['seizure'] == 1].index]
    slow_sz_y = [slow['value'].iloc[i] for i in slow[slow['seizure'] == 1].index]
    monthly_sz_x = [monthly['timestamp'].iloc[i] for i in monthly[monthly['seizure'] == 1].index]
    monthly_sz_y = [monthly['value'].iloc[i] for i in monthly[monthly['seizure'] == 1].index]
    fast_sz_x = [i for i in drop(data, first_time, last_time)[drop(data, first_time, last_time)['seizure'] == 1].timestamp]
    fast_sz_y = [i for i in drop(data, first_time, last_time)[drop(data, first_time, last_time)['seizure'] == 1].MA]
    
    # Plot the slow and monthly smoothed heart rate with seizures
    for dataset, sfilter, szdatax, szdatay in zip([slow, monthly], ['2-day', '7-day'], [slow_sz_x, monthly_sz_x], [slow_sz_y, monthly_sz_y]):
        for colour in ['red', 'darkmagenta', 'lightseagreen']:
            
            plt.style.use('seaborn-white')
            fig = plt.figure(figsize=graph_size)
            plt.plot(dataset['timestamp'], dataset['value'], 'black', linewidth=plot_lw, label = '{} smoothed heart rate'.format(sfilter))
            plt.plot(szdatax, szdatay, c=colour, linestyle='', marker='o', ms=marker_sz, markeredgewidth=0, label='seizures')
            plt.ylabel('Heart Rate (BPM)', fontsize=font_size_tick)
            plt.yticks(fontsize=font_size_tick)
            plt.xticks(rotation=25, fontsize=font_size_tick)
        
            ax = plt.gca()
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            plt.setp(ax.spines.values(), linewidth=ax_lw)
            monthyearFmt = mdates.DateFormatter('%b')
            ax.xaxis.set_major_locator(mdates.MonthLocator())
            ax.xaxis.set_major_formatter(monthyearFmt)

            plt.xlim([dataset.dropna()['timestamp'].iloc[0],dataset.dropna()['timestamp'].iloc[-1]])
            start, end = ax.get_ylim()
            plt.savefig('plots/hr_{}_{}'.format(colour[:3],sfilter) + patient[0:8] + '.png', bbox_inches='tight', dpi=300)
            plt.show()
            
    # plot the fast heart rate with seizures
    plt.style.use('seaborn-white')
    fig = plt.figure(figsize=graph_size)
    plt.plot(drop(data, first_time, last_time)['timestamp'], drop(data, first_time, last_time)['MA'], 'black',
             linewidth=plot_lw, label = '1-hour smoothed heart rate')
    plt.plot(fast_sz_x, fast_sz_y, c='orange', linestyle='', marker='o', ms=marker_sz, markeredgewidth=0, label='seizures')
    plt.ylabel('Heart Rate (BPM)', fontsize=font_size_tick)
    plt.yticks(fontsize=font_size_tick)
    
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.setp(ax.spines.values(), linewidth=ax_lw)
    plt.xlim([first_time,last_time])
    start, end = ax.get_ylim()
    startx, endx = ax.get_xlim()
    plt.xticks([startx,endx],['Day 1', 'Day 14'], rotation=25, fontsize=font_size_tick)
    plt.savefig('plots/hr_inset_' + patient[0:8] + '.png', bbox_inches='tight', dpi=300)
    plt.show()
    
    
    # gather filtered data and plot circular histograms of seizures
    filtered = filtered_data[patient]
    peaks = set([float(i[0]) for i in [i.split('_') for i in filtered.columns if i not in ['timestamp','seizure']]])
    for f in peaks:

        if round(f) == 24:
            bins = 24
            peak_type = 'circadian'
            colour = 'orange'
            dayorhr = 'hour'
            freq = f
            width = (2*np.pi) / bins
        elif f >=(5*24) and f <=(9*24):
            bins = 18
            peak_type = 'weekly'
            colour = 'lightseagreen'
            dayorhr = 'day'
            freq = f/24
            width = (2*np.pi) / bins
        elif f >=(28*24) and f <=(32*24):
            bins = 18
            peak_type = 'monthly'
            colour = 'darkmagenta'
            dayorhr = 'day'
            freq = f/24
            width = (2*np.pi) / bins
        else:
            bins = 18
            colour = 'red'
            if f<16:
                dayorhr = 'hour'
                peak_type = 'ultradian'
                freq = f
            else:
                dayorhr = 'day'
                peak_type = 'multiday'
                freq = f/24
            width = (2*np.pi) / bins
        
        hist,bin_edges = np.histogram(filtered[str(f) + '_phase'], bins = bins, range=(0,np.pi*2))
        
        # Seizure phases
        yseizures = [filtered[str(f) + '_phase'].iloc[i] for i in filtered[filtered['seizure'] == 1].index]
        if not filtered[filtered['seizure'] > 1].empty:
            for j in range(2, int(np.max(filtered[filtered['seizure'] > 1].seizure)) + 1):
                for k in range(j):
                    yseizures += [filtered[str(f) + '_phase'].iloc[i] for i in filtered[filtered['seizure'] == j].index]
        szhist, szbin_edges = np.histogram(yseizures, bins = bins, range=(0,np.pi*2))
        szhist = szhist*max(hist)/max(szhist)
        
        # shuffle distribution
        leads = [np.random.uniform(0,1)*2*np.pi for i in filtered[filtered['seizure'] == 1].index if lead_sz(filtered, i)[0]]
        not_leads = [circ_add(random.choice(leads),(i - lead_sz(filtered, i)[1])*2*np.pi/f) for i in filtered[filtered['seizure'] == 1].index if not lead_sz(filtered, i)[0]]
        p_random_dist = circ_uniformity(np.array(leads + not_leads))

        # get SI index
        SIphases = [np.exp(1j*ph) for ph in yseizures]
        SI=round((1/len(SIphases))*(np.abs(np.sum(SIphases))),2)
        circ_mean = scipy.stats.circmean(yseizures)

        # Check significance of non-uniformity (using Omnibus test)
        pval = circ_uniformity(np.array(yseizures))
        ray = rayleigh(np.array(yseizures))
        if pval <= 0.05/len(peaks): sig = '*'
        else: sig = ''
        
        sig_phase_locking = sig_phase_locking.append({'patient':patient, 'f':f, 'type':peak_type, 'p_omni':pval, 'p_ray':ray, 'si_val':SI, 'circ_mean': circ_mean, 'multi_comparisons': 0.05/len(peaks), 'p_random_dist': p_random_dist}, ignore_index=True)
        
        # check significance with bonferroni test
        if pval <= 0.05/len(peaks):
            bonferroni_sig_cycles = bonferroni_sig_cycles.append({'patient':patient, 'f':f, 'type':peak_type, 'multi_comparisons': 0.05/len(peaks), 'p_omni':pval, 'p_random_dist': p_random_dist, 'si_val':SI, 'circ_mean': circ_mean}, ignore_index=True)

        # circular histogram plot:
        plt.style.use('seaborn-white')
        plt.figure(figsize = hist_size)
        plt.style.use('classic')
        ax = plt.subplot(111, polar=True)
        ax.set_theta_direction(-1)
        ax.bar(bin_edges[:-1], hist, edgecolor='grey', fill=False, width=width, linewidth=ax_lw, label = 'HR phase distribution')
        ax.bar(szbin_edges[:-1], szhist, color=colour, fill=True, alpha=0.7, width=width, \
               linewidth=ax_lw, label='Seizure phase distribution\n{} seizures\np-val = {:.3f}{}\nSI = {}'.format(len(yseizures), pval, sig, SI))
        ax.set_xticks([np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
        ax.set_xticklabels(['Peak', 'Falling', 'Trough', 'Rising'], fontsize = font_size_tick-1, color='grey')
        plt.legend(bbox_to_anchor = (-1,1.5), loc='upper left', fontsize=5)
        ax.set_yticks([])
        ax.tick_params(axis='both', pad=-4)
        ax.grid(False)
        ax.set_theta_zero_location("W")
        plt.setp(ax.spines.values(), linewidth=ax_lw)
        plt.title('{} {} cycle'.format(round(freq,1), dayorhr), fontsize = font_size_tick, pad = 20)
        plt.savefig('plots/hist_' + str(int(f)) + '_' + patient[0:8] + '.png', bbox_inches='tight', dpi=300)
        plt.show()

sig_phase_locking.to_csv('Significant_cycles.csv')
bonferroni_sig_cycles.f = [round(i) for i in bonferroni_sig_cycles.f]
bonferroni_sig_cycles.multi_comparisons = [round_sig(i) for i in bonferroni_sig_cycles.multi_comparisons]
bonferroni_sig_cycles.p_random_dist = [round_sig(float(i)) for i in bonferroni_sig_cycles.p_random_dist]
bonferroni_sig_cycles.p_omni = [round_sig(float(i)) for i in bonferroni_sig_cycles.p_omni]
bonferroni_sig_cycles.si_val = [round_sig(i) for i in bonferroni_sig_cycles.si_val]
bonferroni_sig_cycles.circ_mean = [round_sig(i) for i in bonferroni_sig_cycles.circ_mean]
bonferroni_sig_cycles.to_csv('Bonferroni_significant_cycles.csv')
