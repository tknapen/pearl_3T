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
#     python postprocessing.py sub-$s rl block &
# done

from pearl.surf.surf_draw import av_surf_across_sjs
from pearl.utils.roi import fit_FIR_roi
from pearl.utils.utils import natural_sort
from pearl.utils.plot import plot_deco_results

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

sjs_info = pd.read_csv(os.path.join(os.path.split(preprocessed_data_dir)[0], 'Pearl_subjectID.tsv'), delimiter = '\t')
new_good_names = np.array(sjs_info['new_names'][sjs_info['good_bad']=='good'])

for roi in ['fusifor' , 'temporal_middle']: # , 'temporal_middle'
    all_deco_files = [os.path.join(os.path.split(opd)[0], ngn, 'roi', phase, roi + '_deco.tsv') for ngn in new_good_names]
    all_deco_files = [af for af in all_deco_files if os.path.isfile(af)]
    plot_deco_results(all_deco_files, analysis_info['deconvolution_interval'], output_filename = op.join(opd, roi + '_deco.pdf'))
pl.show()
# for i in range(1,5):
#     for hemi in ['rh', 'lh']:
#         in_surf_files = natural_sort(glob.glob(os.path.join(os.path.split(opd)[0], '*', 'surf', phase, 'zstat%i_flirt-%s.average.mgz'%(i, hemi))))
#         av_surf_across_sjs(in_surf_files, os.path.join(opd, 'surf', 'zstat%i_flirt-%s.average.mgz'%(i, hemi)))