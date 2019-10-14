import numpy as np
import mne
import joblib # for mne multithreading
import os
from os import listdir
from os.path import isfile, join

"""
This is a script for coregistration manually 
"""

# pointers to our directories
rawdir = '/imaging/ai05/phono_oddball/maxfilt_raws'  # raw fifs to input
mne_save_dir = '/imaging/ai05/phono_oddball/mne_files'  # where to save MNE MEG files
source_dir = '/imaging/ai05/phono_oddball/mne_source_models'  # where to save MNE source recon files
struct_dir = '/imaging/ai05/phono_oddball/structurals_renamed'  # where our structs are
fs_sub_dir = '/imaging/ai05/phono_oddball/fs_subdir'  # fresurfer subject dir

flist = [f for f in os.listdir(rawdir) if os.path.isfile(os.path.join(rawdir, f))]
subnames_only = list(set([x.split('_')[0] for x in flist])) # get a unique list of IDs
# get all participants who have a fs_dir
fs_dir_all = os.listdir(fs_sub_dir)
fs_dir_subs = [f for f in fs_dir_all if f in subnames_only]

for file in flist:
    f_only = os.path.basename(file).split('_')  # get filename parts seperated by _
    num = f_only[0]
    full_f = os.path.join(rawdir, file)
    if num in fs_dir_subs: # if participant has source-recon
        mne.gui.coregistration(inst=full_f, subject=num, subjects_dir=fs_sub_dir, advanced_rendering=False)
    else:
        mne.gui.coregistration(inst=full_f, subjects_dir=fs_sub_dir,advanced_rendering=False)
