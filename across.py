# from nipype import config
# config.enable_debug_mode()
# Importing necessary packages
import os
import sys
import os.path as op
import glob
import json
import nipype
import matplotlib.pyplot as pl
import seaborn as sn
import pandas as pd
import numpy as np
from IPython import embed as shell

#
#   run as in:
#
# for s in {001..049}
# do
#     echo sub-$s
#     python fir_fit.py sub-$s rl block &
# done

from pearl.surf.surf_draw import av_surf_across_sjs
from pearl.utils.roi import fit_FIR_roi
from pearl.utils.utils import natural_sort

# the subject id and experiment vars are commandline arguments to this script.
sub_id = 'all'
experiment = 'rl'
phase = str(sys.argv[1])

# from pearl.parameters import *
execfile('pearl/parameters.py')
try:
    os.makedirs(os.path.join(opd, 'surf'))
    os.makedirs(opd)
except:
    pass

# shell()

sjs_info = pd.read_csv(os.path.join(os.path.split(preprocessed_data_dir)[0], 'Pearl_subjectID.tsv'), delimiter = '\t')
new_good_names = np.array(sjs_info['new_names'][sjs_info['good_bad']=='good'])

for roi in ['fusifor']: # , 'temporal_middle'
    all_deco_files = [os.path.join(os.path.split(opd)[0], ngn, 'roi', phase, roi + '_deco.tsv') for ngn in new_good_names]
    all_deco_files = [af for af in all_deco_files if os.path.isfile(af)]
    all_data = np.array([np.loadtxt(df) for df in all_deco_files])
    timepoints = np.linspace(-3,15,all_data.shape[-1])

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

    sets = [[0,3,6], [1,4,7], [9], [10,11]]
    all_event_names = np.array([
                'AB.gain', 'AB.Q_chosen', 'AB.Q_notchosen', 
                'CD.gain', 'CD.Q_chosen', 'CD.Q_notchosen', 
                'EF.gain', 'EF.Q_chosen', 'EF.Q_notchosen', 
                'fb.gain', 'fb.RPE', 'fb.unsRPE' ])

    # sets = [[0,2,4], [1,3,5], [6], [7,8]]
    # all_event_names = np.array([
    #                 'AB.gain', 'AB.Q_diff', 
    #                 'CD.gain', 'CD.Q_diff', 
    #                 'EF.gain', 'EF.Q_diff', 
    #                 'fb.gain', 'fb.RPE', 'fb.unsRPE' ])

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
        sn.tsplot(all_data.transpose((0,2,1))[:,:,sets[x]], time = timepoints, condition = all_event_names[sets[x]], legend = True, ax = s)
        # else: # the difference curves
        #     abcdef_corrs = all_data.transpose((0,2,1))[:,:,sets[x]]
        #     abcdef_corr_diffs = np.array([abcdef_corrs[:,:,(c*2)]-abcdef_corrs[:,:,(c*2)+1] for c in range(3)]).transpose((1,2,0))

        #     sn.tsplot(abcdef_corr_diffs, time = timepoints, condition = ['AB','CD','EF'], legend = True, ax = s)

        pl.legend()
        sn.despine(offset = 10, ax = s)

    pl.tight_layout()

    pl.savefig(op.join(opd, roi + '_deco.pdf'))
pl.show()

# for i in range(1,5):
#     for hemi in ['rh', 'lh']:
#         in_surf_files = natural_sort(glob.glob(os.path.join(os.path.split(opd)[0], '*', 'surf', phase, 'zstat%i_flirt-%s.average.mgz'%(i, hemi))))
#         av_surf_across_sjs(in_surf_files, os.path.join(opd, 'surf', 'zstat%i_flirt-%s.average.mgz'%(i, hemi)))