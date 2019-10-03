import mne
import os
import numpy as np
import joblib


def preprocess_multiple(flist, indir, outdir, overwrite):
    """
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
    for i in range(len(flist)):
        savedfile = __preprocess_individual(os.path.join(indir, flist[i]), outdir, overwrite=overwrite)
        saved_files.append(savedfile)

    return saved_files

def preprocess_multiple_parallel(flist, indir, outdir, overwrite, threads):
    """
    Parameters
    ----------
    Parallel version of the multiple. It uses joblib and 'loky' backend

    :param flist:
        A list of files we want to read in and pre-process
    :param indir:
        Where we find the files
    :param outdir:
        where we want to save those files
    :param overwrite:
        truee or false. whether to overwrite the files if already exist
    :param threads:
        how many threads we should limit this code to run on
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
    for i in range(len(flist)):
        savedfile = __preprocess_individual(os.path.join(indir, flist[i]), outdir, overwrite=overwrite)
        saved_files.append(savedfile)

    return saved_files

def __preprocess_individual(file, outdir, overwrite):
    """
    :param file:
        input file along with path
    :param outdir:
        where we want to save this
    :return: save_file_path:
        a path to the saved and filtered file

    """
    save_file_path = ""
    raw = mne.io.read_raw_fif(file, preload=True)
    f_only = os.path.basename(file).split('_')  # get filename parts seperated by _
    num = f_only[0]

    # check if file exists, if not overwrite then skip and return path
    if os.path.isfile(f'{outdir}/{num}_{f_only[2]}_clean_raw.fif'):
        if not overwrite:
            print(f'file for {num} run {f_only[2]} already exists, skipping to next')
            save_file_path = f'{outdir}/{num}_{f_only[2]}_clean_raw.fif'
            return save_file_path


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

    # if we have identified something !
    if np.any([abs(i) >= 0.2 for i in eog_scores]):
        ica.exclude.extend(eog_inds[0:3])

    # if we can't do this manually by plotting
    else:
        # TODO: Practically this should be moved to after all autodetect is done. This way we don't waste time waiting for
        # user input
        ica.plot_components(inst=raw)
        print('There is no components automatically corellated with blinks')
        print('This is usually because the EOG electrode is bad so select components manually:')
        man_inds = list()
        numb = int(input("Enter how many components to get rid of:"))
        print('Enter each index (remember 1 = 0 in Python):')
        for i in range(int(numb)):
            n = input("num :")
            man_inds.append(int(n))
        ica.exclude.extend(man_inds)

    # now we do this with hearbeat

    ecg_epochs = mne.preprocessing.create_ecg_epochs(raw)  # get epochs of eog (if this exists)
    ecg_inds, ecg_scores = ica.find_bads_ecg(ecg_epochs, threshold=0.5)  # try and find correlated components

    # if one component reaches above threshold then remove components automagically
    if len(ecg_inds) > 0:
        ica.exclude.extend(ecg_inds[0:3])  # exclude top 3 components
        ica.apply(inst=raw)  # apply to raw

    else:  # flag for manual ICA inspection and removal
        # TODO: Practically this should be moved to after all autodetect is done. This way we don't waste time waiting for
        # user input
        ica.plot_components(inst=raw)
        print('There is no components automatically corellated with heartbeat')
        print('This is usually because the ECG electrode is bad so select components manually:')
        man_inds = list()
        numb = int(input("Enter how many components to get rid of:"))
        print('Enter each index (remember 1 = 0 in Python):')
        for i in range(int(numb)):
            n = input("num :")
            man_inds.append(int(n))
        ica.exclude.extend(man_inds)
        ica.apply(inst=raw)

    # save the file
    raw.save(f'{outdir}/{num}_{f_only[2]}_clean_raw.fif', overwrite=overwrite)
    save_file_path = f'{outdir}/{num}_{f_only[2]}_clean_raw.fif'
    # return
    return save_file_path
