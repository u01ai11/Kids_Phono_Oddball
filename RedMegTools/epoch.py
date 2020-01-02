import mne
import os
import numpy as np
import joblib

def epoch_multiple(flist, indir, outdir, keys, trigchan, backup_trigchan,times, overwrite, njobs, offset):
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
    :param backup_trigchan:
        if no events found, the second channel to look for events in
    :param times:
        list of two times to index from
    :param overwrite:
        truee or false. whether to overwrite the files if already exist
    :param njobs:
        the number of jobs for parallel/batch processing
    :param offset:
        how much offset should be applied to triggers
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
            savedfile = __epoch_individual(os.path.join(indir, flist[i]), outdir, keys, trigchan, backup_trigchan, times, overwrite, offset)
            saved_files.append(savedfile)
    if njobs > 1:

        saved_files = joblib.Parallel(n_jobs =njobs)(
            joblib.delayed(__epoch_individual)(os.path.join(indir, thisF), outdir, keys, trigchan, backup_trigchan, times, overwrite, offset) for thisF in flist)

    return saved_files

def __epoch_individual(file, outdir, keys, trigchan , backup_trigchan, times ,overwrite, offset):
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

    f_only = os.path.basename(file).split('_')  # get filename parts seperated by _
    num = f_only[0]

    if not os.path.isfile(file): # not a file
        print(file + ' is not a file')
        save_file_path = file
        return save_file_path

    # check if file exists, if not overwrite then skip and return path
    # TODO: write this so it partial matches and looks for _noecg and _noeog flags in raw data
    if os.path.isfile(f'{outdir}/{num}_{f_only[1]}_epo.fif'):
        if not overwrite:
            print(f'file for {num} run {f_only[1]} already exists, skipping to next')
            save_file_path = f'{outdir}/{num}_{f_only[1]}_epo.fif'
            return save_file_path

    raw = mne.io.read_raw_fif(file, preload=True)

    try:
        events = mne.find_events(raw, shortest_event=1)  # find events
    except ValueError:
        print(f'{num} looks like there is a couple of short events filter them')
        events = mne.find_events(raw, min_duration=1.1/raw.info['sfreq'])

    if len(events) < 5:
        events = mne.find_events(raw, stim_channel=backup_trigchan, shortest_event=1)
        trigchan = backup_trigchan # reassign trig chan


    if offset > 0:
        events[:,0] = events[:,0] + offset

    picks = mne.pick_types(raw.info, meg=True, eog=True, ecg=True, include=trigchan, exclude='bads')  # select channels
    tmin, tmax = times[0], times[1]
    #make epochs from picks and events
    epochs = mne.Epochs(raw, events, keys, tmin, tmax, picks=picks, baseline=(None, 0), preload=True)
    #downsample and decimate epochs
    epochs = mne.Epochs.decimate(epochs, 1)  # downsample to 10hz
    #save
    epochs.save(f'{outdir}/{num}_{f_only[1]}_epo.fif')
    save_file_path = f'{outdir}/{num}_{f_only[1]}_epo.fif'
    # return
    return save_file_path

def evoked_multiple(flist, indir, outdir, keys, contlist, contlist2, overwrite, njobs):
    """ Creates a series of evoked files for each particpant
    Parameters
    ----------
    :param flist:
        A list of files we want to read in and pre-process
    :param indir:
        Where we find the files
    :param outdir:
        where we want to save those files
    :param keys:
        As in epoch, these are the keys for trigger values we want.
        Must match those from epoch
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
            savedfile = __evoked_individual(os.path.join(indir, flist[i]), outdir, keys, contlist, contlist2, overwrite)
            saved_files.append(savedfile)
    if njobs > 1:

        saved_files = joblib.Parallel(n_jobs =njobs)(
            joblib.delayed(__evoked_individual)(os.path.join(indir, thisF), outdir, keys, contlist, contlist2,  overwrite) for thisF in flist)

    return saved_files


def __evoked_individual(file, outdir, keys, contlist, contlist2, overwrite):
    """ Internal function to make evoked from raw files
    :param file:
        input file along with path
    :param outdir:
        where we want to save this
    :param keys:
        As in epoch, these are the keys for trigger values we want.
        Must match those from epoch
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

    if not os.path.isfile(file): # not a file
        print(file + ' is not a file')
        save_file_path = file
        return save_file_path


    # check if output file(s) exist, if not overwrite then skip and return path
    # TODO: this will only check for main average file, might be some situations we want to check all
    if os.path.isfile(f'{outdir}/{num}_{f_only[1]}_ave.fif'):
        if not overwrite:
            print(f'file for {num} run {f_only[1]} already exists, skipping to next')
            save_file_path = f'{outdir}/{num}_{f_only[1]}_ave.fif'
            return save_file_path

    # this will be a list of fnames
    saved_file_path = []
    # read in the evoked file
    try:
        epochs = mne.read_epochs(file)
    except TypeError:
        print('failed to read ' + file)
        saved_file_path = ''
        return saved_file_path

    evoked = epochs.average()
    ev_file_path = f'{outdir}/{num}_{f_only[1]}_ave.fif'
    evoked.save(ev_file_path) # save main average
    saved_file_path.append(ev_file_path) # append to list to store

    # split up by epoch type using key values
    keys_keys = [i for i in keys.keys()]
    evokeds = [epochs[name].average() for name in keys_keys]

    conts = list(contlist.items())
    evoked_cs = [] # blank list
    # calculate contrasts
    for i in range(len(conts)):
        # get name and contast index/weightings
        c_name, cons = conts[i]
        in_args = []
        # loop through and start building input arguments
        # TODO: WARNING: this get's hacky find a better way to pass sign into combine_evoked
        for ii in range(len(cons)):
            if cons[ii] < 0:
                in_args.append(f' -evokeds[{abs(cons[ii])}]')
            else:
                in_args.append(f' evokeds[{abs(cons[ii])}]')

        in_args = ",".join(in_args)
        full_arg = f"evoked_cs.append(mne.combine_evoked([{in_args}], weights='equal'))"
        exec(full_arg)

    # second level subtractions
    conts2 = list(contlist2.items())
    evoked_cs2 = []  # blank list
    # calculate contrasts
    for i in range(len(conts2)):
        # get name and contast index/weightings
        c_name, cons = conts2[i]
        in_args = []
        # loop through and start building input arguments
        # TODO: WARNING: this get's hacky find a better way to pass sign into combine_evoked
        for ii in range(len(cons)):
            if cons[ii] < 0:
                in_args.append(f' -evoked_cs[{abs(cons[ii])}]')
            else:
                in_args.append(f' evoked_cs[{abs(cons[ii])}]')

        in_args = ",".join(in_args)
        full_arg = f"evoked_cs2.append(mne.combine_evoked([{in_args}], weights='equal'))"
        exec(full_arg)

    # loop through contrasts to save them with file names
    for i in range(len(evoked_cs)):
        tosave = f'{outdir}/{num}_{f_only[1]}_{conts[i][0]}_ave.fif'
        evoked_cs[i].save(tosave)
        saved_file_path.append(tosave)

    return saved_file_path
