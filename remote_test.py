import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import mne
import os
from os import listdir
from os.path import isfile, join

import mne_preprocess
#%% set up the folder
rawdir = '/imaging/ai05/phono_oddball/maxfilt_raws'  # raw fifs
flist = [f for f in listdir(rawdir) if isfile(join(rawdir, f))]

#%% test plotting is working on remote session
plt.plot([1, 2, 3, 4])
plt.ylabel('some numbers')
plt.show()
#%% TODO: Sandbox to test, fix later
tmpfile = join(rawdir, flist[1])
raw = mne.io.read_raw_fif(tmpfile)

picks = mne.pick_channels_regexp(raw.ch_names, regexp='MEG')
raw.plot(order=picks, n_channels=len(picks))

#%%
mne_preprocess.process_multiple()
