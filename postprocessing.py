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
from IPython import embed as shell

#
#   run as in:
#
# for s in {001..049}
# do
#     echo sub-$s
#     python postprocessing.py sub-$s rl block &
# done
# python across.py block

from pearl.surf.surf_draw import all2surf
from pearl.rl.roi import fit_FIR_roi_block
from pearl.utils.utils import natural_sort

# the subject id and experiment vars are commandline arguments to this script.
sub_id = str(sys.argv[1])
experiment = str(sys.argv[2])
phase = str(sys.argv[3])

# from pearl.parameters import *
execfile('pearl/parameters.py')

behavior_files = natural_sort(glob.glob(op.join(opd, 'behavior', '*%s*.tsv'%phase)))
in_files = natural_sort(glob.glob(op.join(opd, 'psc', '*%s*.nii.gz'%phase)))
volreg_files = natural_sort(glob.glob(op.join(opd, 'mcf', 'parameter_info', '*%s*.1D'%phase)))

try: 
    os.makedirs(op.join(opd, 'roi', phase))
    os.makedirs(op.join(opd, 'surf', phase))
    os.makedirs(op.join(opd, 'figs', phase))
except:
    pass

if phase == 'block':
    fit_FIR_roi_block(experiment = 'rl',
                    h5_file = op.join(opd, 'h5', 'roi.h5'),
                    in_files = in_files,
                    vol_regressor_list = volreg_files, 
                    behavior_file_list = behavior_files, 
                    mapper_file = 'zstat2_flirt',
                    mask_threshold = analysis_info['stat_mask_threshold'],
                    mask_direction = 'pos',
                    fmri_data_type = 'psc',
                    fir_frequency = analysis_info['deconvolution_frequency'],
                    fir_interval = analysis_info['deconvolution_interval'],
                    roi_list = ['fusifor','temporal_middle', 'calcarine'],
                    output_pdf_dir = op.join(opd, 'figs', phase),
                    output_tsv_dir = op.join(opd, 'roi', phase)
                    )


# taking mapper stats to surface

# all_deriv_nii_files = [op.join(opd, 'mapper_stat', 'zstat%i_flirt.nii.gz'%i) for i in range(1,5)]

# all2surf(
#     all_deriv_nii_files = all_deriv_nii_files, 
#     surf_folder = op.join(opd, 'surf', phase), 
#     FS_subject_dir = FS_subject_dir, 
#     fs_id = sub_id, 
#     reg = op.join(opd, 'reg', 'register.dat'), 
#     pfr = None, 
#     target_subject = 'average', 
#     smooth = 3)