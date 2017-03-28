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


def plot_deco_results_train(all_deco_files, subj_data, interval = [-3,15], output_filename = ''):
    import matplotlib.pyplot as pl
    import seaborn as sn
    import pandas as pd
    import numpy as np
    from IPython import embed as shell

    all_data = np.array([np.loadtxt(df) for df in all_deco_files])
    timepoints = np.linspace(interval[0],interval[1],all_data.shape[-1])

    # sets = [[0], [1,2,3], [4], [5,6]]
    # all_event_names = np.array(['stim.gain', 'stim.Q_chosen', 'stim.Q_notchosen', 'stim.Q_diff', 'fb.gain', 'fb.RPE', 'fb.unsRPE'])

    # sets = [[0], [1], [2], [3,4]]
    # all_event_names = np.array(['stim.gain', 'stim.Q_diff', 'fb.gain', 'fb.RPE', 'fb.unsRPE'])

    # all_event_names = np.array(['stim.gain', 'stim.Q_chosen', 'stim.Q_notchosen', 'fb.gain', 'fb.RPE', 'fb.unsRPE', 'fb.Q_chosen', 'fb.Q_notchosen' ])

    # sets = [[0,3,6], [1,2,4,5,7,8], [9], [10,11]]
    # all_event_names = np.array([
    #             'AB.gain', 'AB.Q_chosen', 'AB.Q_notchosen', 
    #             'CD.gain', 'CD.Q_chosen', 'CD.Q_notchosen', 
    #             'EF.gain', 'EF.Q_chosen', 'EF.Q_notchosen', 
    #             'fb.gain', 'fb.RPE', 'fb.unsRPE' ])

    # sets = [[0,2,4], [1,3,5], [6], [7,8]]
    # all_event_names = np.array([
    #                 'AB.gain', 'AB.Q_diff', 
    #                 'CD.gain', 'CD.Q_diff', 
    #                 'EF.gain', 'EF.Q_diff', 
    #                 'fb.gain', 'fb.RPE', 'fb.unsRPE' ])

    # shell()

    # offset the values at times < 0
    # gauge_times = timepoints < 0
    # gauged_data = all_data - all_data[:,:,gauge_times].mean(axis = -1)[:,:,np.newaxis]
    # all_data = gauged_data

    sets = [[0,3,6], [1,4,7], [9], [10,11]]
    all_event_names = np.array([
                'AB.gain', 'AB.Q_chosen', 'AB.Q_notchosen', 
                'CD.gain', 'CD.Q_chosen', 'CD.Q_notchosen', 
                'EF.gain', 'EF.Q_chosen', 'EF.Q_notchosen', 
                'fb.gain', 'fb.RPE', 'fb.unsRPE' ])
    colors = np.array(['r','r', 'r','g','g','g','b','b','b','k','c','m'])

    color_dict = dict(zip(all_event_names, colors))

    sn.set_style('ticks')
    f = pl.figure(figsize = (9,8))

    for x in range(len(sets)):
        s = f.add_subplot(2,2,x+1)
        s.axhline(0, c='k', lw = 0.25)
        s.axvline(0, c='k', lw = 0.25)
        s.set_xlabel('Time [s]')
        if x % 2 == 0:
            s.set_ylabel('BOLD % signal change')
        else:
            s.set_ylabel('beta values')

        # if x != 1:
        sn.tsplot(all_data.transpose((0,2,1))[:,:,sets[x]], time = timepoints, condition = all_event_names[sets[x]], legend = True, ax = s, color = color_dict)
        # else: # the difference curves
        #     abcdef_corrs = all_data.transpose((0,2,1))[:,:,sets[x]]
        #     abcdef_corr_diffs = np.array([abcdef_corrs[:,:,(c*2)]-abcdef_corrs[:,:,(c*2)+1] for c in range(3)]).transpose((1,2,0))

        #     sn.tsplot(abcdef_corr_diffs, time = timepoints, condition = ['AB','CD','EF'], legend = True, ax = s)
        min_d = all_data.transpose((0,2,1))[:,:,sets[x]].min()

        plot_significance_lines(all_data.transpose((0,2,1))[:,:,sets[x]], time_points = timepoints, offset=min_d/20.0, slope=min_d/30.0, p_value_cutoff = 0.05, pal = colors[sets[x]])
        pl.legend()
        sn.despine(offset = 10, ax = s)

    pl.tight_layout()

    pl.savefig(output_filename)


def plot_deco_results_test(all_deco_files, subj_data, interval = [-3,15], output_filename = ''):
    import matplotlib.pyplot as pl
    import seaborn as sn
    import pandas as pd
    import numpy as np
    from IPython import embed as shell

    all_data = np.array([np.loadtxt(df) for df in all_deco_files])
    timepoints = np.linspace(interval[0],interval[1],all_data.shape[-1])

    cond_diffs = ['ll-ww', 'll-wl_u', 'ww-wl_u']

    roi_data_diffs = np.squeeze(np.array([  roi_data[:,:,conditions=='ll'] - roi_data[:,:,conditions=='ww'], 
                            roi_data[:,:,conditions=='ll'] - roi_data[:,:,conditions=='wl_u'], 
                            roi_data[:,:,conditions=='ww'] - roi_data[:,:,conditions=='wl_u'] ]
                            )).transpose(1,2,0)

    sets = [[0,3,6], [1,4,7], [9], [10,11]]
    all_event_names = np.array([
                'AB.gain', 'AB.Q_chosen', 'AB.Q_notchosen', 
                'CD.gain', 'CD.Q_chosen', 'CD.Q_notchosen', 
                'EF.gain', 'EF.Q_chosen', 'EF.Q_notchosen', 
                'fb.gain', 'fb.RPE', 'fb.unsRPE' ])
    colors = np.array(['r','r', 'r','g','g','g','b','b','b','k','c','m'])

    color_dict = dict(zip(all_event_names, colors))

    sn.set_style('ticks')
    f = pl.figure(figsize = (9,8))

    for x in range(len(sets)):
        s = f.add_subplot(2,2,x+1)
        s.axhline(0, c='k', lw = 0.25)
        s.axvline(0, c='k', lw = 0.25)
        s.set_xlabel('Time [s]')
        if x % 2 == 0:
            s.set_ylabel('BOLD % signal change')
        else:
            s.set_ylabel('beta values')

        # if x != 1:
        sn.tsplot(all_data.transpose((0,2,1))[:,:,sets[x]], time = timepoints, condition = all_event_names[sets[x]], legend = True, ax = s, color = color_dict)
        # else: # the difference curves
        #     abcdef_corrs = all_data.transpose((0,2,1))[:,:,sets[x]]
        #     abcdef_corr_diffs = np.array([abcdef_corrs[:,:,(c*2)]-abcdef_corrs[:,:,(c*2)+1] for c in range(3)]).transpose((1,2,0))

        #     sn.tsplot(abcdef_corr_diffs, time = timepoints, condition = ['AB','CD','EF'], legend = True, ax = s)
        min_d = all_data.transpose((0,2,1))[:,:,sets[x]].min()

        plot_significance_lines(all_data.transpose((0,2,1))[:,:,sets[x]], time_points = timepoints, offset=min_d/20.0, slope=min_d/30.0, p_value_cutoff = 0.05, pal = colors[sets[x]])
        pl.legend()
        sn.despine(offset = 10, ax = s)

    pl.tight_layout()

    pl.savefig(output_filename)
