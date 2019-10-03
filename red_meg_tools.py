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

    # check if file exists, if not overwrite then skip and return path
    if os.path.isfile(f'{outdir}/{num}_{f_only[2]}_clean_raw.fif'):
        if not overwrite:
            print(f'file for {num} run {f_only[2]} already exists, skipping to next')
            save_file_path = f'{outdir}/{num}_{f_only[2]}_clean_raw.fif'
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

    # now we do this with hearbeat

    ecg_epochs = mne.preprocessing.create_ecg_epochs(raw)  # get epochs of eog (if this exists)
    ecg_inds, ecg_scores = ica.find_bads_ecg(ecg_epochs, threshold=0.5)  # try and find correlated components

    # if one component reaches above threshold then remove components automagically
    if len(ecg_inds) > 0:
        ica.exclude.extend(ecg_inds[0:3])  # exclude top 3 components
        ica.apply(inst=raw)  # apply to raw

    else:  # flag for manual ICA inspection and removal
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

def epoch_multiple(flist, indir, outdir, keys, trigchan, times, overwrite, njobs):
    """ Epochs a list of files using the passed in arguments.
    Parameters
    ----------
    :param flist:
        A list of files we want to read in and pre-process
    :param indir:
        Where we find the files
    :param outdir:
        where we want to save those files
    :param keys:
        dictionary pairs of labels and trigger numbers for events
    :param trigchan:
        the channel to look for trigger from
    :param times:
        list of two times to index from
    :param overwrite:
        truee or false. whether to overwrite the files if already exist
    :param njobs:
        the number of jobs for parallel/batch processing
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
            savedfile = __epoch_individual(os.path.join(indir, flist[i]), outdir, keys, trigchan, times, overwrite)
            saved_files.append(savedfile)
    if njobs > 1:

        saved_files = joblib.Parallel(n_jobs =njobs)(
            joblib.delayed(__epoch_individual)(os.path.join(indir, thisF), outdir, keys, trigchan, times, overwrite) for thisF in flist)

    return saved_files

def __epoch_individual(file, outdir, keys, trigchan , times ,overwrite):
    """ Internal function to make epochs from raw files
    :param file:
        input file along with path
    :param outdir:
        where we want to save this
    :return: save_file_path:
        a path to the saved and filtered file
    :param keys:
        dictionary pairs of labels and trigger numbers for events
    :param trigchan:
        the channel to look for trigger from
    :param times:
        list of two times to index from
    :param overwrite:
        truee or false. whether to overwrite the files if already exist

    """
    raw = mne.io.read_raw_fif(file, preload=True)
    f_only = os.path.basename(file).split('_')  # get filename parts seperated by _
    num = f_only[0]

    # check if file exists, if not overwrite then skip and return path
    # TODO: write this so it partial matches and looks for _noecg and _noeog flags in raw data
    if os.path.isfile(f'{outdir}/{num}_{f_only[2]}_clean_raw.fif'):
        if not overwrite:
            print(f'file for {num} run {f_only[2]} already exists, skipping to next')
            save_file_path = f'{outdir}/{num}_{f_only[2]}_clean_raw.fif'
            return save_file_path

    events = mne.find_events(raw)  # find events
    picks = mne.pick_types(raw.info, meg=True, eog=True, ecg=True, include=trigchan, exclude='bads')  # select channels
    tmin, tmax = times[0], times[1]
    #make epochs from picks and events
    epochs = mne.Epochs(raw, events, keys, tmin, tmax, picks=picks, baseline=(None, 0), preload=True)
    #downsample and decimate epochs
    epochs = mne.Epochs.decimate(epochs, 1)  # downsample to 10hz
    #save
    epochs.save(f'{outdir}/{num}_{f_only[2]}_epo.fif')
    save_file_path = f'{outdir}/{num}_{f_only[2]}_epo.fif'
    # return
    return save_file_path