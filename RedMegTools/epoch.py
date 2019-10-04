import mne
import os
import numpy as np
import joblib

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
    if os.path.isfile(f'{outdir}/{num}_{f_only[2]}_epo.fif'):
        if not overwrite:
            print(f'file for {num} run {f_only[2]} already exists, skipping to next')
            save_file_path = f'{outdir}/{num}_{f_only[2]}_epo.fif'
            return save_file_path

    try:
        events = mne.find_events(raw)  # find events
    except ValueError:
        print(f'{num} looks like there is a couple of short events filter them')
        events = mne.find_events(raw, min_duration=1.1/raw.info['sfreq'])



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

def evoked_multiple(flist, indir, outdir, contlist, contlist2, overwrite, njobs):
    """ Creates a series of evoked files for each particpant
    Parameters
    ----------
    :param flist:
        A list of files we want to read in and pre-process
    :param indir:
        Where we find the files
    :param outdir:
        where we want to save those files
    :param contlist:
        an ordered dict for contrasts to make with combine evoked
        format e.g.
        contlist = {
            'MNN': [0, -1, -2],
            'MNN Word': [0, -1],
            'MNN Non-Word': [0, -2]
        }
        each key is a contrast label, the values are a list of indices, with the sign
        refering to the waiting of each item.
        These indices match onto the keys we passed through in the epoching stage
    :param contlist2:
        another ordered dict describing contrasts of the contrasts to be made
        e.g.
        contlist2 = {
            'MNN_diff' = [1, -2]
        }
        this time the indices refer to the position in contlist
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
            savedfile = __evoked_individual(os.path.join(indir, flist[i]), outdir, contlist, contlist_2, overwrite)
            saved_files.append(savedfile)
    if njobs > 1:

        saved_files = joblib.Parallel(n_jobs =njobs)(
            joblib.delayed(__evoked_individual)(os.path.join(indir, thisF), outdir, contlist, contlist_2,  overwrite) for thisF in flist)

    return saved_files

def __evoked_individual(file, outdir, contlist, contlist2, overwrite):
    """ Internal function to make evoked from raw files
    :param file:
        input file along with path
    :param outdir:
        where we want to save thi
    :param contlist:
        an ordered dict for contrasts to make with combine evoked
        format e.g.
        contlist = {
            'MNN': [0, -1, -2],
            'MNN Word': [0, -1],
            'MNN Non-Word': [0, -2]
        }
        each key is a contrast label, the values are a list of indices, with the sign
        refering to the waiting of each item.
        These indices match onto the keys we passed through in the epoching stage
    :param contlist2:
        another ordered dict describing contrasts of the contrasts to be made
        e.g.
        contlist2 = {
            'MNN_diff' = [1, -2]
        }
        this time the indices refer to the position in contlist
    :param overwrite:
        truee or false. whether to overwrite the files if already exis
    :return: save_file_path:
        a path to the saved and filtered file

    """
    f_only = os.path.basename(file).split('_')  # get filename parts seperated by _
    num = f_only[0]

    # check if output file(s) exist, if not overwrite then skip and return path
    # TODO: this will only check for main average file, might be some situations we want to check all
    if os.path.isfile(f'{outdir}/{num}_{f_only[2]}_ave.fif'):
        if not overwrite:
            print(f'file for {num} run {f_only[2]} already exists, skipping to next')
            save_file_path = f'{outdir}/{num}_{f_only[2]}_ave.fif'
            return save_file_path

    # read in the evoked file
    epochs = mne.read_epochs(file)
    evoked = epochs.average()
    save_file_path = f'{outdir}/{num}_{f_only[2]}_ave.fif'

    evokeds = [epochs[name].average() for name in ('Freq', 'Dev Word', 'Dev Non-Word')]

    save_file_path = f'{outdir}/{num}_{f_only[2]}_ave.fif'
    # return
    return save_file_path