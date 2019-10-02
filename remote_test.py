import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import mne
import joblib # for mne multithreading
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
raw.filter(1, None, l_trans_bandwidth='auto', filter_length='auto',
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
    # TODO: Practically this should be moved to after all autodetect is done. This way we don't waste time waiting for
    # user input
    ica.plot_components(inst=raw)
    print('There is no components automatically corellated with blinks')
    print('This is usually because the EOG electrode is bad so select components manually:')
    man_inds = list()
    num = int(input("Enter how many components to get rid of:"))
    print('Enter each index (remember 1 = 0 in Python):')
    for i in range(int(num)):
        n = input("num :")
        man_inds.append(int(n))
    ica.exclude.extend(man_inds)
    ica.apply(inst=raw)

#%% Construct Source Model -- FreeSurfer
struct_dir = '/imaging/ai05/phono_oddball/structurals_renamed' # TODO: this dir to be passed in
fs_sub_dir = '/imaging/ai05/phono_oddball/fs_subdir' # TODO: this also needs to be parsed in
if os.path.isdir(fs_sub_dir):
    os.system(f"tcsh -c 'freesurfer_6.0.0 && setenv SUBJECTS_DIR {fs_sub_dir}'")
else:
    os.system(f"tcsh -c 'freesurfer_6.0.0 && mkdir {fs_sub_dir} && setenv SUBJECTS_DIR {fs_sub_dir}'")

f_only = os.path.basename(tmpfile).split('_') # get filename parts seperated by _
num = f_only[0] # first in this list is the participant id
T1_name = num + '_T1w.nii.gz'

if os.path.isfile(struct_dir + '/' + T1_name):
    os.system(f"tcsh -c 'recon-all -i {struct_dir}/{T1_name} -s {num} -all -parallel'")
else:
    print('no T1 found for ' + num)

#%% Source-space
#TODO: We are using Edwin's brain for now, this needs to be changed later
fs_sub = 'edwin_2019' # subname
# compute source space
src_space = mne.setup_source_space(fs_sub, spacing='oct6', surface='white', subjects_dir=fs_sub_dir, n_jobs=2)
mne.write_source_spaces(fs_sub_dir+'/'+ fs_sub + '/'+ fs_sub+'-oct6-src.fif', src_space) # write to freesurfer dir
# compute Boundary Element Model
# use os.system to run this in tcsh shell setup freesurfer and subject dir as well
os.system(f"tcsh -c 'freesurfer_6.0.0 && setenv SUBJECTS_DIR && mne watershed_bem -s {fs_sub} -d {fs_sub_dir}'")
os.system(term_command)





#%%
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
