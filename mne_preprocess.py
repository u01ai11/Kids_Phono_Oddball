import mne
import os


def process_multiple(flist, fmapping, indir, outdir):
    # Takes two inputs files and there relative subject mapping and reads them in
    # returns a list of pointers in MNE format
    for i in range(len(flist)):
        __process_individual(os.path.join(indir, flist(i)), fmapping(i))


def __process_individual(file, subname):
    # Private function to read in files and process
    raw = mne.io.read_raw_fif(file) # read raw (not in memory)
    # detect bad channels
    picks = mne.pick_channels_regexp(raw.ch_names, regexp='MEG')
    raw.plot(order=picks, n_channels=len(picks))
