import mne
import os
import numpy as np
import joblib

def preprocess_multiple(flist, indir, outdir, overwrite, njobs):
    """ Takes a list of raw files and preprocesses them
    Parameters
    ----------
    :param flist:
        A list of files we want to read in and pre-process
    :param indir:
        Where we find the files
    :param outdir:
        where we want to save those files
    :param overwrite:
        truee or false. whether to overwrite the files if already exist
    :return saved_files:
        A list of files we have saved
    """

    # first check if indir and outdir exist
    # if not outdoor make it
    # if not indir raise error
    if not os.path.isdir(indir):
        raise Exception(f'path {indir} does not exist, edit and try again')
    if not os.path.isdir(outdir):
        os.mkdir(outdir)

    saved_files = []

    if njobs == 1:
        for i in range(len(flist)):
            savedfile = __preprocess_individual(os.path.join(indir, flist[i]), outdir, overwrite=overwrite)
            saved_files.append(savedfile)
    if njobs > 1:

        saved_files = joblib.Parallel(n_jobs =njobs)(
            joblib.delayed(__preprocess_individual)(os.path.join(indir, thisF), outdir, overwrite) for thisF in flist)

    return saved_files


def __preprocess_individual(file, outdir, overwrite):
    """ Internal function for preprocessing raw MEG files
    :param file:
        input file along with path
    :param outdir:
        where we want to save this
    :return: save_file_path:
        a path to the saved and filtered file

    """
    save_file_path = ""

    f_only = os.path.basename(file).split('_')  # get filename parts seperated by _
    num = f_only[0]

    # check if any of these files exists, if not overwrite then skip and return path
    # could be any of these names
    check_fnames = [f'{outdir}/{num}_{f_only[2]}_noeog_noecg_clean_raw.fif',
                    f'{outdir}/{num}_{f_only[2]}_noecg_clean_raw.fif',
                    f'{outdir}/{num}_{f_only[2]}_noeog_clean_raw.fif',
                    f'{outdir}/{num}_{f_only[2]}_clean_raw.fif']

    if np.any([os.path.isfile(f) for f in check_fnames]):
        index = np.where([os.path.isfile(f) for f in check_fnames])[0]
        if not overwrite:
            print(f'file for {num} run {f_only[2]} already exists, skipping to next')
            save_file_path = check_fnames[index[0]]
            return save_file_path

    # read file
    raw = mne.io.read_raw_fif(file, preload=True)

    # 50 Hz remove power line noise with a notch filter
    raw.notch_filter(np.arange(50, 241, 50), filter_length='auto',
                     phase='zero')

    # 1Hz highpass filter to remove slow drift (might have to revisit this as ICA works better with 1Hz hp)
    raw.filter(1, None, l_trans_bandwidth='auto', filter_length='auto',
               phase='zero')

    # Run ICA on raw data to find blinks and eog
    ica = mne.preprocessing.ICA(n_components=25, method='infomax').fit(
        raw)
    try:
        # look for and remove EOG
        eog_epochs = mne.preprocessing.create_eog_epochs(raw)  # get epochs of eog (if this exists)
        eog_inds, eog_scores = ica.find_bads_eog(eog_epochs, threshold=1)  # try and find correlated components

        # define flags for tracking if we found components matching or not
        no_ecg_removed = False
        no_eog_removed = False

        # if we have identified something !
        if np.any([abs(i) >= 0.2 for i in eog_scores]):
            ica.exclude.extend(eog_inds[0:3])

        else:
            print(f'{num} run {f_only[2]} cannot detect eog automatically manual ICA must be done')
            no_eog_removed = True

    except RuntimeError:
        print(f'{num} run {f_only[2]} cannot detect eog automatically manual ICA must be done')
        no_eog_removed = True

    # now we do this with hearbeat
    try:
        ecg_epochs = mne.preprocessing.create_ecg_epochs(raw)  # get epochs of eog (if this exists)
        ecg_inds, ecg_scores = ica.find_bads_ecg(ecg_epochs, threshold=0.5)  # try and find correlated components

        # if one component reaches above threshold then remove components automagically
        if len(ecg_inds) > 0:
            ica.exclude.extend(ecg_inds[0:3])  # exclude top 3 components
            ica.apply(inst=raw)  # apply to raw

        else:  # flag for manual ICA inspection and removal
            print(f'{num} run {f_only[2]} cannot detect ecg automatically manual ICA must be done')
            no_ecg_removed = True
    except RuntimeError:
        print(f'{num} run {f_only[2]} cannot detect ecg automatically manual ICA must be done')
        no_ecg_removed = True

    # save the file
    if no_ecg_removed and no_eog_removed:
        outfname = f'{outdir}/{num}_{f_only[2]}_noeog_noecg_clean_raw.fif'
    elif no_ecg_removed:
        outfname = f'{outdir}/{num}_{f_only[2]}_noecg_clean_raw.fif'
    elif no_eog_removed:
        outfname = f'{outdir}/{num}_{f_only[2]}_noeog_clean_raw.fif'
    else:
        outfname = f'{outdir}/{num}_{f_only[2]}_clean_raw.fif'

    raw.save(outfname, overwrite=overwrite)
    save_file_path = outfname
    # return
    return save_file_path
