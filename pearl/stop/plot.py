#  function for plotting significance regions
def plot_significance_lines(data, time_points, offset, slope, p_value_cutoff = 0.05, pal = None):
    """plot_significance_lines takes , or regions, or something, and calculates cluster-based p-values against 0.
        data numpy.array, subjects by time by conditions
        offset float, offset in y position for lines in plot
        slope float, change in y position for consecutive lines in plot
        p_value_cutoff float, below which p_value to plot.
    """
    import matplotlib.pyplot as pl
    import numpy as np
    import seaborn as sn
    import mne

    if pal is None:
        pal = sn.dark_palette("green", data.shape[-1])

    for i in range(data.shape[-1]): # loop across regions
        clusters = mne.stats.permutation_cluster_1samp_test(data[...,i])
        for cluster_times, p_val in zip (clusters[1], clusters[2]):
            if p_val < p_value_cutoff:
                s = np.arange(time_points.shape[0])[cluster_times]
                pl.plot([time_points[s[0]], time_points[s[-1]]], [offset + slope * i, offset + slope * i], c = pal[i], linewidth = 3.0, alpha = 0.8)


def plot_deco_results(all_deco_files, subj_data, roi_name, interval = [-3,15], output_filename = ''):
    import matplotlib.pyplot as pl
    import seaborn as sn
    import pandas as pd
    import numpy as np
    from IPython import embed as shell
    import statsmodels.api as sm


    all_data = np.array([np.loadtxt(df) for df in all_deco_files])
    timepoints = np.linspace(interval[0],interval[1],all_data.shape[-1])

    sets = [[0],[1],[2]]
    all_event_names = np.array([
                'correct', 'succesful_stop', 'failed_stop' ])
    colors = np.array(['b','g','r'])

    color_dict = dict(zip(all_event_names, colors))

    sn.set_style('ticks')
    f = pl.figure(figsize = (5,11))
    s = f.add_subplot(4,1,1)
    s.set_title(roi_name + ' gain')
    s.axhline(0, c='k', lw = 0.25)
    s.axvline(0, c='k', lw = 0.25)
    s.set_xlabel('Time [s]')
    s.set_ylabel('BOLD % signal change')

    min_d = -0.15
    for x in range(len(sets)):
        sn.tsplot(all_data.transpose((0,2,1))[:,:,sets[x]], time = timepoints, condition = all_event_names[sets[x]], legend = True, ax = s, color = color_dict)
        plot_significance_lines(all_data.transpose((0,2,1))[:,:,sets[x]], time_points = timepoints, offset=min_d/20.0, slope=min_d/30.0, p_value_cutoff = 0.05, pal = colors[sets[x]])
    sn.despine(offset = 10, ax = s)

    pl.legend()

    ##############################################################################################################
    #
    # Now, we compute the correlations with Beta
    #
    #
    ##############################################################################################################
    sig = -np.log10(0.05)
    # shell()

    # X = np.vstack([np.ones(len(all_deco_files)), np.array(subj_data['Beta'], dtype = float), np.array(subj_data['alphaL'], dtype = float), np.array(subj_data['alphaG'], dtype = float)]).T
    beta = np.array(subj_data['Beta'], dtype = float)
    beta = (beta - beta.mean()) / beta.std()

    ssrt = np.array(subj_data['SSRT'], dtype = float)
    ssrt = (ssrt - ssrt.mean()) / ssrt.std()

    X = np.vstack([np.ones(len(all_deco_files)), beta, ssrt]).T

    p_T_vals = np.zeros((len(all_event_names),all_data.shape[-1], 3))
    tcs = np.zeros((len(all_event_names),all_data.shape[-1], 3))
    for i, et in enumerate(all_event_names):
        for x in range(all_data.shape[-1]):
            model = sm.OLS(np.squeeze(all_data[:,i,x]),X)
            results = model.fit()
            p_T_vals[i,x,:2] = -np.log10(results.pvalues[1:])
            p_T_vals[i,x,2] = -np.log10(results.f_pvalue)
            tcs[i,x] = results.params

    for i, c in enumerate(['Beta', 'SSRT']):
        s = f.add_subplot(4,1,2+i)
        s.set_title(roi_name + ' corrs %s'%c)
        s.axhline(0, c='k', lw = 0.25)
        s.axvline(0, c='k', lw = 0.25)
        s.set_xlabel('Time [s]')
        s.set_ylabel('beta values')        
        for j, en in enumerate(all_event_names):
            data = tcs[j,:,i+1]
            pl.plot(timepoints, data, color = colors[j], label = en)

            sig_periods = p_T_vals[j,:,i] > sig

            nr_blocks = int(np.floor(np.abs(np.diff(sig_periods)).sum() / 2.0))
            if nr_blocks > 0:
                print('# blocks found: %i'%nr_blocks)
                for b in range(nr_blocks):
                    time_sig = np.arange(timepoints.shape[0])[np.r_[False, np.abs(np.diff(sig_periods)) > 0]][b*2:(b*2)+2]
                    # time_sig = np.arange(timepoints.shape[0])[sig_periods]
                    pl.plot([timepoints[time_sig[0]]-0.5, timepoints[time_sig[-1]] + 0.5], [-0.25-0.05*j, -0.25-0.05*j], color = colors[j], linewidth = 3.0, alpha = 0.8)

        pl.legend()
        sn.despine(offset = 10, ax = s)

    pl.tight_layout()
    pl.show()

    pl.savefig(output_filename)
