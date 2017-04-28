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


def plot_deco_results_test(all_deco_files, subj_data, event_conditions, roi, event_conditions_for_covariates, sj_covariates = ['Beta', 'SSRT'], interval = [-3,15], output_filename = '', 
    rl_test_FIR_amplitude_range = [0,0], rl_test_FIR_pe_range = [0,0]):
    import matplotlib.pyplot as pl
    import seaborn as sn
    import pandas as pd
    import numpy as np
    from IPython import embed as shell
    import statsmodels.api as sm

    stats_threshold = 0.05
    all_data = np.array([np.loadtxt(df) for df in all_deco_files])
    timepoints = np.linspace(interval[0],interval[1],all_data.shape[-1])
    new_event_conditions = [evc.replace('.','_') for evc in event_conditions]

    covariate_event_indices = [event_conditions.index(ecc) for ecc in event_conditions_for_covariates]
    cond_diffs = ['ll-ww', 'll-wl_u', 'ww-wl_u']
    all_event_names = new_event_conditions
    # shell()
    sets = [0,1,2,3]
    colors = np.array(['g','orange','orange','r'])

    color_dict = dict(zip(all_event_names, colors))


    roi_data_diffs = np.squeeze(np.array([  all_data[:,new_event_conditions=='ll'] - all_data[:,new_event_conditions=='ww'], 
                            all_data[:,new_event_conditions=='ll'] - all_data[:,new_event_conditions=='wl_u'], 
                            all_data[:,new_event_conditions=='ww'] - all_data[:,new_event_conditions=='wl_u'] ]
                            )).transpose(1,0,2)

    sig = -np.log10(stats_threshold)
    # if roi_name == 'Caudate':
    #     shell()
    sn.set_style('ticks')
    f = pl.figure(figsize = (5,11))
    s = f.add_subplot(3,1,1)
    s.set_title(roi + ' gain')
    s.axhline(0, c='k', lw = 0.25)
    s.axvline(0, c='k', lw = 0.25)
    s.set_xlabel('Time [s]')
    s.set_ylabel('BOLD % signal change')

    min_d = all_data.transpose((0,2,1))[:,:,[0,1,2]].mean(axis = 0).min()
    for x in range(len(sets)):
        sn.tsplot(all_data.transpose((0,2,1))[:,:,sets[x]], time = timepoints, condition = all_event_names[sets[x]], legend = True, ax = s, color = color_dict)
        plot_significance_lines(all_data.transpose((0,2,1))[:,:,[sets[x]]], time_points = timepoints, offset=0.0125+rl_test_FIR_amplitude_range[0], slope=0.025, p_value_cutoff = 0.05, pal = [colors[sets[x]]])
    s.set_ylim(rl_test_FIR_amplitude_range)
    s.set_xlim([timepoints[0], timepoints[-1]])
    sn.despine(offset = 10, ax = s)

    pl.legend()

    ##############################################################################################################
    #
    # Now, we compute the correlations with Beta
    #
    ##############################################################################################################

    # ssrt
    ssrt = np.array(subj_data['SSRT'], dtype = float)
    ssrt = (ssrt - ssrt.mean()) / ssrt.std()

    # beta
    beta = np.array(subj_data['Beta'], dtype = float)
    beta = (beta - beta.mean()) / beta.std()

    X = np.vstack([np.ones(len(all_deco_files)), ssrt, beta]).T

    p_T_vals = np.zeros((len(all_event_names),all_data.shape[-1], X.shape[1]+1))
    tcs = np.zeros((len(all_event_names),all_data.shape[-1], X.shape[1]))
    for i, et in enumerate(all_event_names):
        for x in range(all_data.shape[-1]):
            model = sm.OLS(np.squeeze(all_data[:,i,x]),X)
            results = model.fit()
            p_T_vals[i,x,:X.shape[1]] = -np.log10(results.pvalues)
            p_T_vals[i,x,-1] = -np.log10(results.f_pvalue)
            tcs[i,x] = results.params

    for i, c in enumerate(['SSRT', 'Beta']): # , 'Beta'
        s = f.add_subplot(3,1,2+i)
        s.set_title(roi + ' corrs %s'%c)
        s.axhline(0, c='k', lw = 0.25)
        s.axvline(0, c='k', lw = 0.25)
        s.set_xlabel('Time [s]')
        s.set_ylabel('beta values')        
        for j, en in enumerate(all_event_names):
            data = tcs[j,:,i+1]
            pl.plot(timepoints, data, color = colors[j], label = en)

            sig_periods = p_T_vals[j,:,i+1] > sig

            # take care of start and end of deconvolution interval.
            if sig_periods[0] == True:
                sig_periods[0] = False
            if sig_periods[-1] == True:
                sig_periods[-1] = False

            nr_blocks = int(np.floor(np.abs(np.diff(sig_periods.astype(int))).sum() / 2.0))
            if nr_blocks > 0:
                print('# blocks found: %i'%nr_blocks)
                for b in range(nr_blocks):
                    time_sig = np.arange(timepoints.shape[0])[np.r_[False, np.abs(np.diff(sig_periods)) > 0]][b*2:(b*2)+2]
                    # time_sig = np.arange(timepoints.shape[0])[sig_periods]
                    pl.plot([timepoints[time_sig[0]]-0.5, timepoints[time_sig[-1]] + 0.5], [0.0125 + rl_test_FIR_pe_range[0]+0.0125*j, 0.0125 + rl_test_FIR_pe_range[0]+0.0125*j], color = colors[j], linewidth = 3.0, alpha = 0.8)
        s.set_ylim(rl_test_FIR_pe_range)
        s.set_xlim([timepoints[0], timepoints[-1]])
        pl.legend()
        sn.despine(offset = 10, ax = s)

    pl.tight_layout()

    pl.savefig(output_filename)

def plot_deco_results_test_MR():



    rd_diff = np.squeeze(roi_data[:,:,conditions == 'll']) - np.squeeze(roi_data[:,:,conditions == 'ww'])
    betas, res, rank, sem = np.linalg.lstsq(X, rd_diff)
    bbruns = np.ones((nr_bs, X.shape[-1], rd_diff.shape[-1]))
    for bbr in range(nr_bs):
        inds = np.random.randint(0, X.shape[0], X.shape[0])
        bbs, _res, _rank, _sem = np.linalg.lstsq(X[inds], rd_diff[inds])
        bbruns[bbr] = bbs
    bs_sd = bbruns.std(axis = 0)     

    sig = -np.log10(0.05/8)
    p_T_vals = np.zeros((2, rd_diff.shape[-1], 4))
    for x in range(rd_diff.shape[-1]):
        model = sm.OLS(np.squeeze(rd_diff[:,x]),X)
        results = model.fit()
        p_T_vals[0,x,:3] = -np.log10(results.pvalues)
        p_T_vals[1,x,:3] = results.tvalues

        p_T_vals[0,x,3] = -np.log10(results.f_pvalue)
        p_T_vals[1,x,3] = results.fvalue



    f = pl.figure(figsize = (8,8))
    f.suptitle('STN')

    s = f.add_subplot(111)
    # rd = np.squeeze(roi_data[:,:,conditions == 'll'])
    # betas, res, rank, sem = np.linalg.lstsq(X, rd)

    bbruns = np.ones((nr_bs, X.shape[-1], rd_diff.shape[-1]))
    for bbr in range(nr_bs):
        inds = np.random.randint(0, X.shape[0], X.shape[0])
        bbs, _res, _rank, _sem = np.linalg.lstsq(X[inds], rd_diff[inds])
        bbruns[bbr] = bbs
    bs_sd = bbruns.std(axis = 0)        
    for x in range(betas.shape[0]):
        pl.plot(times, betas[x], colors[x], label = beta_names[x])
        pl.fill_between(times, betas[x] - bs_sd[x], betas[x] + bs_sd[x], color = colors[x], alpha = 0.2)

    # significance
    sig = -np.log10(0.05)
    p_T_vals = np.zeros((2, rd_diff.shape[-1], 4))
    for x in range(rd_diff.shape[-1]):
        model = sm.OLS(np.squeeze(rd_diff[:,x]),X)
        results = model.fit()
        p_T_vals[0,x,:3] = -np.log10(results.pvalues)
        p_T_vals[1,x,:3] = results.tvalues

        p_T_vals[0,x,3] = -np.log10(results.f_pvalue)
        p_T_vals[1,x,3] = results.fvalue

    # shell()
    sig_periods = p_T_vals[0,:,[0,1]] > sig
    for i in [0,1]:
        time_sig = np.arange(times.shape[0])[sig_periods[i]]
        pl.plot([times[time_sig[0]]-0.5, times[time_sig[-1]] + 0.5], [[-0.025, -0.025], [-0.035, -0.035]][i], color = ['red','green'][i], linewidth = 3.0, alpha = 0.8)


    s.set_title('Lose-Lose')
    s.axhline(0, color = 'k', alpha = 0.5, lw = 0.5)
    # s.axvline(0, color = 'k', alpha = 0.5, lw = 0.5)
    pl.legend()
    s.set_ylabel('Beta values')
    # s.set_xlabel('time [s]')
    sn.despine(offset=10)
    s.set_xlim(time_period)
    s.set_xticks([0,5,10])
    s.set_ylim([-0.04,0.06001])

    # shell()

    s = f.add_subplot(2,2,2)
    rd = np.squeeze(roi_data[:,:,conditions == 'ww'])
    betas, res, rank, sem = np.linalg.lstsq(X, rd)

    bbruns = np.ones((nr_bs, X.shape[-1], rd.shape[-1]))
    for bbr in range(nr_bs):
        inds = np.random.randint(0, X.shape[0], X.shape[0])
        bbs, _res, _rank, _sem = np.linalg.lstsq(X[inds], rd[inds])
        bbruns[bbr] = bbs
    bs_sd = bbruns.std(axis = 0)    
    for x in range(betas.shape[0]):
        pl.plot(times, betas[x], colors[x], label = beta_names[x])
        pl.fill_between(times, betas[x] - bs_sd[x], betas[x] + bs_sd[x], color = colors[x], alpha = 0.2)

    # # significance is never reached
    # sig = -np.log10(0.05/8)
    # p_T_vals = np.zeros((2, rd.shape[-1], 4))
    # for x in range(rd.shape[-1]):
    #     model = sm.OLS(np.squeeze(rd[:,x]),X)
    #     results = model.fit()
    #     p_T_vals[0,x,:3] = -np.log10(results.pvalues)
    #     p_T_vals[1,x,:3] = results.tvalues

    #     p_T_vals[0,x,3] = -np.log10(results.f_pvalue)
    #     p_T_vals[1,x,3] = results.fvalue
    # # shell()
    # sig_periods_ww = p_T_vals[0,:,[0,1]] > sig
    # for i in [0,1]:
    #     time_sig = np.arange(times.shape[0])[sig_periods_ww[i]]
    #     pl.plot([times[time_sig[0]]-0.5, times[time_sig[-1]] + 0.5], [[-0.025, -0.025], [-0.035, -0.035]][i], color = ['red','green'][i], linewidth = 3.0, alpha = 0.8)


    s.set_title('Win-Win')
    s.axhline(0, color = 'k', alpha = 0.5, lw = 0.5)
    # s.axvline(0, color = 'k', alpha = 0.5, lw = 0.5)
    pl.legend()
    s.set_ylabel('Beta values')
    s.set_xlabel('time [s]')
    sn.despine(offset=10)
    s.set_xlim(time_period)
    s.set_xticks([0,5,10])
    s.set_ylim([-0.04,0.06001])


    # group split
    evt = 'll'
    s = f.add_subplot(2,2,3)

    group_values = np.array([np.array(ssa.evts['SSRT'])[0] for ssa in self.ssas])
    group_median = np.median(group_values)

    group = group_values <= group_median


    idx = conditions == evt
    this_condition_data = (roi_data[group,:,idx],roi_data[-group,:,idx])

    s.set_title('Lose-Lose - SSRT')
    sn.tsplot(this_condition_data[0], time = times, condition = ['SSRT fast'], ci = 68, color = 'gray', ls = '--')
    sn.tsplot(this_condition_data[1], time = times, condition = ['SSRT slow'], ci = 68, color = 'gray')

    i = 0
    time_sig = np.arange(times.shape[0])[sig_periods[i]]
    pl.plot([times[time_sig[0]]-0.5, times[time_sig[-1]] + 0.5], [[-0.025, -0.025], [-0.035, -0.035]][i], color = ['red','green'][i], linewidth = 3.0, alpha = 0.8)

    s.axhline(0, color = 'k', alpha = 0.5, lw = 0.5)
    # s.axvline(0, color = 'k', alpha = 0.5, lw = 0.5)
    s.set_xlabel('time [s]')
    s.set_ylabel('Z-scored BOLD')
    sn.despine(offset=10)
    s.set_xlim(time_period)
    s.set_xticks([0,5,10])
    s.set_ylim([-0.05,0.09])


    s = f.add_subplot(2,2,4)

    group_values = np.array([np.array(ssa.evts['Beta'])[0] for ssa in self.ssas])
    group_median = np.median(group_values)

    group = group_values <= group_median


    idx = conditions == evt
    this_condition_data = (roi_data[group,:,idx],roi_data[-group,:,idx])

    s.set_title('Lose-Lose - Beta')
    sn.tsplot(this_condition_data[0], time = times, condition = ['Explore'], ci = 68, color = 'gray', ls = '--')
    sn.tsplot(this_condition_data[1], time = times, condition = ['Exploit'], ci = 68, color = 'gray')

    i = 1
    time_sig = np.arange(times.shape[0])[sig_periods[i]]
    pl.plot([times[time_sig[0]]-0.5, times[time_sig[-1]] + 0.5], [[-0.025, -0.025], [-0.025, -0.025]][i], color = ['red','green'][i], linewidth = 3.0, alpha = 0.8)

    s.axhline(0, color = 'k', alpha = 0.5, lw = 0.5)
    # s.axvline(0, color = 'k', alpha = 0.5, lw = 0.5)
    s.set_xlabel('time [s]')
    s.set_ylabel('Z-scored BOLD')
    sn.despine(offset=10)
    s.set_xlim(time_period)
    s.set_xticks([0,5,10])
    s.set_ylim([-0.05,0.09])




    pl.tight_layout()


