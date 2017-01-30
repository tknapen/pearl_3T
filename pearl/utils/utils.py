from __future__ import division, print_function

def convert_edf_2_hdf5(edf_file, low_pass_pupil_f = 6.0, high_pass_pupil_f = 0.01):
    """converts the edf_file to hdf5 using hedfpy
    
    Requires hedfpy

    Parameters
    ----------
    edf_file : str
        absolute path to edf file.
    low_pass_pupil_f : float
        low pass cutoff frequency for band-pass filtering of the pupil signal
    high_pass_pupil_f : float
        high pass cutoff frequency for band-pass filtering of the pupil signal
    Returns
    -------
    hdf5_file : str
        absolute path to hdf5 file.
    """

    import os
    import os.path as op
    from hedfpy import HDFEyeOperator
    import tempfile

    tempdir = tempfile.mkdtemp()
    temp_edf = op.join(tempdir, op.split(edf_file)[-1])

    os.system('cp ' + edf_file + ' ' + temp_edf)

    hdf5_file = op.join(tempdir, op.split(op.splitext(edf_file)[0] + '.h5')[-1])
    alias = op.splitext(op.split(edf_file)[-1])[0]

    ho = HDFEyeOperator(hdf5_file)
    ho.add_edf_file(temp_edf)
    ho.edf_message_data_to_hdf(alias = alias)
    ho.edf_gaze_data_to_hdf(alias = alias, pupil_hp = high_pass_pupil_f, pupil_lp = low_pass_pupil_f)

    return hdf5_file

def combine_eye_hdfs_to_nii_hdf(nii_hdf5_file, eye_hdf_filelist, new_alias = 'eye'):
    import os.path as op
    import tables as tb

    nii_hf = tb.open_file(nii_hdf5_file, 'a') 
    eye_hfs = [tb.open_file(ef, 'r') for ef in eye_hdf_filelist]
    eye_group_names = [op.splitext(op.split(ef)[-1])[0] for ef in eye_hdf_filelist]

    try:
        nii_group = nii_hf.get_node("/", name = new_alias, classname='Group')
    except tb.NoSuchNodeError:
        print('Adding group ' + new_alias + ' to ' + nii_hdf5_file)
        nii_group = nii_hf.create_group("/", new_alias, new_alias)

    for ef, en in zip(eye_hfs, eye_group_names):
        ef.copy_node(where = '/' + en, newparent = nii_group, newname = en, overwrite = True, recursive = True)
        ef.close()

    nii_hf.close()

    return nii_hdf5_file

def mask_nii_2_hdf5(in_files, mask_files, hdf5_file, folder_alias):
    """masks data in in_files with masks in mask_files,
    to be stored in an hdf5 file

    Takes a list of 3D or 4D fMRI nifti-files and masks the
    data with all masks in the list of nifti-files mask_files.
    These files are assumed to represent the same space, i.e.
    that of the functional acquisitions. 
    These are saved in hdf5_file, in the folder folder_alias.

    Parameters
    ----------
    in_files : list
        list of absolute path to functional nifti-files.
        all nifti files are assumed to have the same ndim
    mask_files : list
        list of absolute path to mask nifti-files.
        mask_files are assumed to be 3D
    hdf5_file : str
    	absolute path to hdf5 file.
   	folder_alias : str
   		name of the to-be-created folder in the hdf5 file.

    Returns
    -------
    hdf5_file : str
        absolute path to hdf5 file.
    """

    import nibabel as nib
    import os.path as op
    import numpy as np
    import tables

    success = True

    mask_data = [np.array(nib.load(mf).get_data(), dtype = bool) for mf in mask_files]
    nifti_data = [nib.load(nf).get_data() for nf in in_files]

    mask_names = [op.split(mf)[-1].split('_vol.nii.gz')[0] for mf in mask_files]
    nifti_names = [op.split(nf)[-1].split('.nii.gz')[0] for nf in in_files]

    h5file = tables.open_file(hdf5_file, mode = "a", title = hdf5_file)
    # get or make group for alias folder
    try:
        folder_alias_run_group = h5file.get_node("/", name = folder_alias, classname='Group')
    except tables.NoSuchNodeError:
        print('Adding group ' + folder_alias + ' to this file')
        folder_alias_run_group = h5file.create_group("/", folder_alias, folder_alias)

    for (roi, roi_name) in zip(mask_data, mask_names):
        # get or make group for alias/roi
        try:
            run_group = h5file.get_node(where = "/" + folder_alias, name = roi_name, classname='Group')
        except tables.NoSuchNodeError:
            print('Adding group ' + folder_alias + '_' + roi_name + ' to this file')
            run_group = h5file.create_group("/" + folder_alias, roi_name, folder_alias + '_' + roi_name)

        h5file.create_array(run_group, roi_name, roi, roi_name + ' mask file for reconstituting nii data from masked data')

        for (nii_d, nii_name) in zip(nifti_data, nifti_names):
            print('roi: %s, nifti: %s'%(roi_name, nii_name))
            n_dims = len(nii_d.shape)
            if n_dims == 3:
                these_roi_data = nii_d[roi]
            elif n_dims == 4:   # timeseries data, last dimension is time.
                these_roi_data = nii_d[roi,:]
            else:
                print("n_dims in data {nifti} do not fit with mask".format(nii_name))
                success = False

            h5file.create_array(run_group, nii_name, these_roi_data, roi_name + ' data from ' + nii_name)

    h5file.close()

    return hdf5_file

def roi_data_from_hdf(data_types_wildcards, roi_name_wildcard, hdf5_file, folder_alias):
    """takes data_type data from masks stored in hdf5_file

    Takes a list of 4D fMRI nifti-files and masks the
    data with all masks in the list of nifti-files mask_files.
    These files are assumed to represent the same space, i.e.
    that of the functional acquisitions. 
    These are saved in hdf5_file, in the folder folder_alias.

    Parameters
    ----------
    data_types_wildcards : list
        list of data types to be loaded.
        correspond to nifti_names in mask_2_hdf5
    roi_name_wildcard : str
        wildcard for masks. 
        corresponds to mask_name in mask_2_hdf5.
    hdf5_file : str
        absolute path to hdf5 file.
    folder_alias : str
        name of the folder in the hdf5 file from which data
        should be loaded.

    Returns
    -------
    output_data : list
        list of numpy arrays corresponding to data_types and roi_name_wildcards
    """
    import tables
    import itertools
    import fnmatch
    import numpy as np

    h5file = tables.open_file(hdf5_file, mode = "r")

    try:
        folder_alias_run_group = h5file.get_node(where = '/', name = folder_alias, classname='Group')
    except NoSuchNodeError:
        # import actual data
        print('No group ' + folder_alias + ' in this file')
        # return None


    all_roi_names = h5file.list_nodes(where = '/' + folder_alias, classname = 'Group')
    roi_names = [rn._v_name for rn in all_roi_names if roi_name_wildcard in rn._v_name]
    if len(roi_names) == 0:
        print('No rois corresponding to ' + roi_name_wildcard + ' in group ' + folder_alias)
        # return None
    
    data_arrays = []
    for roi_name in roi_names:
        try:
            roi_node = h5file.get_node(where = '/' + folder_alias, name = roi_name, classname='Group')
        except tables.NoSuchNodeError:
            print('No data corresponding to ' + roi_name + ' in group ' + folder_alias)
            pass
        all_data_array_names = h5file.list_nodes(where = '/' + folder_alias + '/' + roi_name)
        data_array_names = [adan._v_name for adan in all_data_array_names]
        selected_data_array_names = list(itertools.chain(*[fnmatch.filter(data_array_names, dtwc) for dtwc in data_types_wildcards]))
        
        # if sort_data_types:
        selected_data_array_names = sorted(selected_data_array_names)
        if len(data_array_names) == 0:
            print('No data corresponding to ' + str(selected_data_array_names) + ' in group /' + folder_alias + '/' + roi_name)
            pass
        else:
            print('Taking data corresponding to ' + str(selected_data_array_names) + ' from group /' + folder_alias + '/' + roi_name)
            data_arrays.append([])
            for dan in selected_data_array_names:
                data_arrays[-1].append(eval('roi_node.__getattr__("' + dan + '").read()'))

            data_arrays[-1] = np.hstack(data_arrays[-1]) # stack across timepoints or other values per voxel
    all_roi_data_np = np.vstack(data_arrays)    # stack across regions to create a single array of voxels by values (i.e. timepoints)

    h5file.close()

    return all_roi_data_np

def leave_one_out_lists(input_list):
    """leave_one_out_lists takes creates a list of lists, with each element
    of the input_list left out of the returned lists once, in order.


    Parameters
    ----------
    input_list : list
        list of items, for instance absolute paths to nii files

    Returns
    -------
    output_data : list
        list of lists
    """

    out_lists = []
    for x in input_list:
        out_lists.append([y for y in input_list if y != x])

    return out_lists

def suff_file_name(in_file, suff = '_av_loo', extension = '.nii.gz' ):
    out_file = in_file[:-len(extension)] + suff + extension

    return out_file

def combine_cv_prf_fit_results_all_runs(basedir, fit_dir, nr_slices = 51):
    """combine_fit_results_one_fold combines a per-slice
    
    Requires hedfpy

    Parameters
    ----------
    basedir : str
        absolute path to directory in which to recursively search for the input
        files to the fits that happened on cartesius.
    fit_dir : str
        absolute path to directory in which to find the fit result files.
    nr_slices : int (default: 51)
        number of slices in nii files

    Returns
    -------
    output_files : list
        absolute paths to output nifti files.
    """   

    import os
    import glob

    all_files = glob.glob(os.path.join(basedir, '*.nii.gz'))
    loo_files = glob.glob(os.path.join(basedir, 'loo', '*.nii.gz'))

    all_files.extend(loo_files)
    all_files_no_ext = [os.path.join(fit_dir, os.path.split(afl)[-1][:-7]) for afl in all_files]

    output_files = []
    for basefilename in all_files_no_ext:
        output_files.append(
            combine_cv_prf_fit_results_one_fold(basefilename = basefilename, nr_slices = nr_slices)
            )

    return output_files

def combine_cv_prf_fit_results_one_fold(basefilename, nr_slices = 51):
    """combine_fit_results_one_fold combines a per-slice
    

    Parameters
    ----------
    basefilename : str
        absolute path to stem of per-slice files.
    nr_slices : int (default: 51)
        number of slices in nii files

    Returns
    -------
    output_file : str
        absolute path to output nifti file.
    """    
    import os
    import nibabel as nb
    import glob
    import numpy as np

    in_files = sorted(glob.glob(basefilename + '*.nii.gz'))
    output_file = os.path.join(basefilename + '_est_all.nii.gz')

    print('creating ' + output_file)

    all_fit_files = [basefilename + '_est_%s.nii.gz'%str(i).zfill(2) for i in range(nr_slices)]

    nif = nb.load(all_fit_files[0])
    all_fits = np.zeros(nif.get_data().shape)

    for i, aff in enumerate(all_fit_files):
        nif = nb.load(aff)
        all_fits[:,:,i,:] = nif.get_data()[:,:,i,:]

    img = nb.Nifti1Image(all_fits, affine=nif.affine, header=nif.header)
    img.to_filename(output_file)

    return output_file

def natural_sort(l):
    import re
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(l, key = alphanum_key)