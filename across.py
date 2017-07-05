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
#     python postprocessing.py sub-$s rl test &
# done

from pearl.surf.surf_draw import av_surf_across_sjs
from pearl.utils.utils import natural_sort
import pearl.rl as rl
import pearl.stop as stop

# the subject id and experiment vars are commandline arguments to this script.
sub_id = 'all'
experiment = str(sys.argv[1])
phase = str(sys.argv[2])

# from pearl.parameters import *
# execfile('pearl/parameters.py')
exec(open("pearl/parameters.py").read())


try:
    os.makedirs(os.path.join(opd, 'surf'))
    os.makedirs(opd)
except:
    pass

# shell()

sjs_info = pd.read_csv(os.path.join(raw_data_dir, 'participants.tsv'), delimiter = '\t')
if experiment == 'stop':
    which_sjs = (sjs_info['Incl_ex'] == 'ok')
    new_good_names = np.array(sjs_info['participant_id'][which_sjs])
    good_sjs_info = sjs_info[which_sjs] 
elif experiment == 'rl':
    which_sjs = (sjs_info['Incl_ex'] == 'ok')
    new_good_names = np.array(sjs_info['participant_id'][which_sjs])
    good_sjs_info = sjs_info[which_sjs]

print(len(new_good_names))
print(new_good_names)

if phase == 'train' and experiment == 'rl':
    for roi in analysis_info['rl_train_rois']: # , 'temporal_middle'
        all_deco_files = [os.path.join(os.path.split(opd)[0], ngn, 'roi', phase, roi + '_deco_train.tsv') for ngn in new_good_names]
        all_deco_files = [af for af in all_deco_files if os.path.isfile(af)]
        rl.plot.plot_deco_results_train(all_deco_files, good_sjs_info, analysis_info['deconvolution_interval'], output_filename = op.join(opd, roi + '_deco.pdf'))

if phase == 'test' and experiment == 'rl':

    # extraction of significantly correlated timepoints
    # sj_covariates = ['Beta','SSRT']
    # roi='maxSTN25exc_flirt'
    # all_deco_files = [os.path.join(os.path.split(opd)[0], ngn, 'roi', phase, roi + '_deco_test_projection.tsv') for ngn in new_good_names]
    # all_deco_files = [af for af in all_deco_files if os.path.isfile(af)]
    # rl.plot.extract_timepoints_results_test(all_deco_files=all_deco_files, 
    #                                         subj_data=good_sjs_info, 
    #                                         event_conditions=analysis_info['rl_test_event_conditions'],
    #                                         roi=roi, 
    #                                         event_conditions_for_covariates=analysis_info['rl_test_event_types_for_covariates'],
    #                                         sj_covariates=sj_covariates,
    #                                         interval=analysis_info['deconvolution_interval'])

    for roi in analysis_info['rl_test_rois']: # , 'temporal_middle'
        which_signal_selection = 'projection'
        # for sj_covariates in [['Beta','SSRT'],['ac_ww','ac_ll'], ['medRT_ww','medRT_ll']]:
        #     all_deco_files = [os.path.join(os.path.split(opd)[0], ngn, 'roi', phase, roi + '_deco_test_%s.tsv'%which_signal_selection) for ngn in new_good_names]
        #     all_deco_files = [af for af in all_deco_files if os.path.isfile(af)]
        #     rl.plot.plot_deco_results_test(all_deco_files=all_deco_files, 
        #                                     subj_data=good_sjs_info, 
        #                                     event_conditions=analysis_info['rl_test_event_conditions'],
        #                                     roi=roi, 
        #                                     event_conditions_for_covariates=analysis_info['rl_test_event_types_for_covariates'],
        #                                     sj_covariates=sj_covariates, 
        #                                     interval=analysis_info['deconvolution_interval'], 
        #                                     output_filename = op.join(opd, roi + '_deco_%s.pdf'%'_'.join(sj_covariates)),
        #                                     rl_test_FIR_amplitude_range = analysis_info['rl_test_FIR_amplitude_range'], rl_test_FIR_pe_range = analysis_info['rl_test_FIR_pe_range'])

        sj_covariates_dicts = [
                {
                'ww': ['SSRT', 'ac_ww'],
                'll': ['SSRT', 'ac_ll'],
                'wl_u': ['SSRT', 'ac_wlu'],
                # 'wl_l': ['SSRT', 'ac_wll'],
                },
                {
                'ww': ['SSRT', 'Beta', 'ac_ww'],
                'll': ['SSRT', 'Beta', 'ac_ll'],
                'wl_u': ['SSRT', 'Beta', 'ac_wlu'],
                # 'wl_l': ['SSRT', 'Beta', 'ac_wll'],
                },
                {
                'ww': ['SSRT', 'Beta'],
                'll': ['SSRT', 'Beta'],
                'wl_u': ['SSRT', 'Beta'],
                # 'wl_l': ['SSRT', 'Beta', 'ac_wll'],
                },
                {
                'ww': ['SSRT', 'Beta','medRT_ww'],
                'll': ['SSRT', 'Beta','medRT_ll'],
                'wl_u': ['SSRT', 'Beta','medRT_wlu'],
                # 'wl_l': ['SSRT', 'Beta', 'ac_wll'],
                }

        ]
        for sj_covariates in sj_covariates_dicts:
            fn_suffix = '_'.join(['_'.join(sj_covariates[sck]) for sck in sj_covariates.keys()])
            all_deco_files = [os.path.join(os.path.split(opd)[0], ngn, 'roi', phase, roi + '_deco_test_%s.tsv'%which_signal_selection) for ngn in new_good_names]
            all_deco_files = [af for af in all_deco_files if os.path.isfile(af)]
            rl.plot.plot_deco_results_test_covariates_per_event_type(all_deco_files=all_deco_files, 
                                            subj_data=good_sjs_info, 
                                            event_conditions=analysis_info['rl_test_event_conditions'],
                                            roi=roi, 
                                            event_conditions_for_covariates=analysis_info['rl_test_event_types_for_covariates'],
                                            sj_covariates=sj_covariates, 
                                            interval=analysis_info['deconvolution_interval'], 
                                            output_filename = op.join(opd, roi + '_deco_%s.pdf'%fn_suffix),
                                            rl_test_FIR_amplitude_range = analysis_info['rl_test_FIR_amplitude_range'], rl_test_FIR_pe_range = analysis_info['rl_test_FIR_pe_range'])


if experiment == 'stop':
    sj_covariates_dicts = [
        {
        'correct': ['SSRT'],
        'succesful_stop': ['SSRT'],
        'Failed_stop': ['SSRT'],
        # 'wl_l': ['SSRT', 'ac_wsuccesful_stop'],
        },
        {
        'correct': ['SSRT', 'Beta'],
        'succesful_stop': ['SSRT', 'Beta'],
        'Failed_stop': ['SSRT', 'Beta'],
        # 'wl_l': ['SSRT', 'Beta', 'ac_wsuccesful_stop'],
        }
    ]
    for roi in analysis_info['stop_rois']: # , 'temporal_middle'
        for sj_covariates in sj_covariates_dicts:
            fn_suffix = '_'.join(['_'.join(sj_covariates[sck]) for sck in sj_covariates.keys()])
            all_deco_files = [os.path.join(os.path.split(opd)[0], ngn, 'roi', phase, roi + '_deco_stop.tsv') for ngn in new_good_names]
            all_deco_files = [af for af in all_deco_files if os.path.isfile(af)]
            stop.plot.plot_deco_results_covariates_per_event_type(all_deco_files, 
                    good_sjs_info, roi, analysis_info['deconvolution_interval'],
                    output_filename = op.join(opd, roi + '_deco%s.pdf'%fn_suffix),
                    sj_covariates = sj_covariates,
                    stop_FIR_amplitude_range = analysis_info['stop_FIR_amplitude_range'], 
                    stop_FIR_pe_range = analysis_info['stop_FIR_pe_range'])

# pl.show()


# for i in range(1,5):
#     for hemi in ['rh', 'lh']:
#         in_surf_files = natural_sort(glob.glob(os.path.join(os.path.split(opd)[0], '*', 'surf', phase, 'zstat%i_flirt-%s.average.mgz'%(i, hemi))))
#         av_surf_across_sjs(in_surf_files, os.path.join(opd, 'surf', 'zstat%i_flirt-%s.average.mgz'%(i, hemi)))