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

#%% TODO: Sandbox to test, when working place in mne_preprocess module function
tmpfile = join(rawdir, flist[1])
raw = mne.io.read_raw_fif(tmpfile, preload=True)
raw_or = mne.io.read_raw_fif(tmpfile, preload=True) # original version for comparison
# plot this to visualise raw data
picks = mne.pick_channels_regexp(raw.ch_names, regexp='MEG.')
raw.plot(order=picks, n_channels=10, start=30)

# 50 Hz remove power line noise with a notch filter
raw.notch_filter(np.arange(50, 241, 50), picks=picks, filter_length='auto',
                 phase='zero')

# 0.1Hz highpass filter to remove slow drift
raw.filter(0.1, None, l_trans_bandwidth='auto', filter_length='auto',
           phase='zero')

# ICA to detect EOG (blinks)
ica = mne.preprocessing.ICA(n_components=.99, method='infomax').fit(raw)  # run ica on raw data
eog_epochs = mne.preprocessing.create_eog_epochs(raw)  # get epochs of eog (if this exists)
eog_inds, eog_scores = ica.find_bads_eog(eog_epochs, threshold=1)  # try and find correlated components

# if one component reaches above threshold then remove components automagically
if np.any([abs(i) >= 0.3 for i in eog_scores]):
    ica.exclude.extend(eog_inds[0:3])  # exclude top 3 components
    ica.apply(inst=raw)  # apply to raw

else: # flag for manual ICA inspection and removal
    ica.plot_components(inst=raw)
    print('There is no components automatically corellated with blinks')
    print('This is usually because the EOG electrode is bad so select components manually')




# Extract before downsampling to avoid precision errors
events = mne.find_events(raw) # find events from file
event_id = {'Freq': 10, 'Dev Word': 11, 'Dev Non-Word': 12}  # trigger codes for events
trig_chan = 'STI101_up' # name of the chanel to take values from
picks = mne.pick_types(raw.info, meg=True, eog=True, ecg=True, include=trig_chan, exclude='bads')  # select channels
tmin, tmax = -0.2, 0.8
epochs = mne.Epochs(raw, events, event_id, tmin, tmax, picks=picks, baseline=(None, 0), preload=True)

# downsample and decimate  epochs
epochs = mne.Epochs.decimate(epochs, 1) # downsample to 10hz

# ICA and
mne.viz.plot_raw_psd(raw)
#%%
#mne_preprocess.process_multiple()
