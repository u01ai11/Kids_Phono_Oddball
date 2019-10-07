import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import mne
import joblib # for mne multithreading
import os
from os import listdir
from os.path import isfile, join

"""
TEST SCRIPT 

This is more of a sandbox script where I can get the pipeline working.

Once this all works on one subject I will then move this into different modules and tie in a group analysis using..

thise modules 
"""
import mne_preprocess
#%% set up the folder
rawdir = '/imaging/ai05/phono_oddball/maxfilt_raws'  # raw fifs
flist = [f for f in listdir(rawdir) if isfile(join(rawdir, f))]
mne_save_dir = '/imaging/ai05/phono_oddball/mne_files'
#%% TODO: Sandbox to test, when working place in mne_preprocess module function
#%% read in and filter
tmpfile = join(rawdir, flist[43])
f_only = os.path.basename(tmpfile).split('_') # get filename parts seperated by _
num = f_only[0]

raw = mne.io.read_raw_fif(tmpfile, preload=True)
#raw_or = mne.io.read_raw_fif(tmpfile, preload=True) # original version for comparison
# plot this to visualise raw data
picks = mne.pick_channels_regexp(raw.ch_names, regexp='MEG.')
#raw.plot(order=picks, n_channels=10, start=30)

# 50 Hz remove power line noise with a notch filter
raw.notch_filter(np.arange(50, 241, 50), picks=picks, filter_length='auto',
                 phase='zero')

# 1Hz highpass filter to remove slow drift (might have to revisit this as ICA works better with 1Hz hp)
raw.filter(1, None, l_trans_bandwidth='auto', filter_length='auto',
           phase='zero')

#%% ICA to detect EOG (blinks)
ica = mne.preprocessing.ICA(n_components=25, method='infomax').fit(raw)  # run ica on raw data, currently takes ~10 min
eog_epochs = mne.preprocessing.create_eog_epochs(raw)  # get epochs of eog (if this exists)
eog_inds, eog_scores = ica.find_bads_eog(eog_epochs, threshold=1)  # try and find correlated components

# if one component reaches above threshold then remove components automagically
if np.any([abs(i) >= 0.2 for i in eog_scores]):
    ica.exclude.extend(eog_inds[0:3])  # exclude top 3 components
    # ica.apply(inst=raw)  # apply to raw (wait until ecg done also)
else: # flag for manual ICA inspection and removal
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
    # ica.exclude.extend(man_inds) # wait for ecg also


#%% Detect ecg on the same ICA
ecg_epochs = mne.preprocessing.create_ecg_epochs(raw)  # get epochs of eog (if this exists)
ecg_inds, ecg_scores = ica.find_bads_ecg(ecg_epochs, threshold=0.5)  # try and find correlated components

# if one component reaches above threshold then remove components automagically
if len(ecg_inds) > 0:
    ica.exclude.extend(ecg_inds[0:3])  # exclude top 3 components
    ica.apply(inst=raw)  # apply to raw

else: # flag for manual ICA inspection and removal
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

#%% Save the filtered and ICA cleaned raw data
raw.save(f'{mne_save_dir}/{num}_{f_only[2]}_clean_raw.fif')
# raw = mne.io.read_raw_fif(f'{mne_save_dir}/{num}_{f_only[2]}_clean_raw.fif') # if needed
#%% EPOCHS
# Extract before downsampling to avoid precision errors
events = mne.find_events(raw) # find events from file
event_id = {'Freq': 10, 'Dev Word': 11, 'Dev Non-Word': 12}  # trigger codes for events
trig_chan = 'STI101_up' # name of the chanel to take values from
picks = mne.pick_types(raw.info, meg=True, eog=True, ecg=True, include=trig_chan, exclude='bads')  # select channels
tmin, tmax = -0.2, 0.8
epochs = mne.Epochs(raw, events, event_id, tmin, tmax, picks=picks, baseline=(None, 0), preload=True)

# downsample and decimate  epochs
epochs = mne.Epochs.decimate(epochs, 1) # downsample to 10hz
epochs.save(f'{mne_save_dir}/{num}_{f_only[2]}_epo.fif')
#epochs = mne.read_epochs(f'{mne_save_dir}/{num}_{f_only[2]}_epo.fif') # if needed

#%% EVOKED
evoked = epochs.average();
evoked.save(f'{mne_save_dir}/{num}_{f_only[2]}_ave.fif')
# TODO: check the read function, it returns a list for some reason
# evoked = mne.read_evokeds(f'{mne_save_dir}/{num}_{f_only[2]}_ave.fif') # if needed

#multiple conditions
evokeds = [epochs[name].average() for name in ('Freq', 'Dev Word', 'Dev Non-Word')]

#%% Construct Source Model -- FreeSurfer
struct_dir = '/imaging/ai05/phono_oddball/structurals_renamed'  # TODO: this dir to be passed in
fs_sub_dir = '/imaging/ai05/phono_oddball/fs_subdir'  # TODO: this also needs to be passed in
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

#%% Source-space & BEM Watershed
# TODO: We are using one person, add this into input in module in the future
source_dir = '/imaging/ai05/phono_oddball/mne_source_models' # dir to save models to
fs_sub = num  # subname
struct_dir = '/imaging/ai05/phono_oddball/structurals_renamed'  # defined above but may not be the case in module
fs_sub_dir = '/imaging/ai05/phono_oddball/fs_subdir'  # so lets define it again

# compute source space
src_space = mne.setup_source_space(fs_sub, spacing='oct6', surface='white', subjects_dir=fs_sub_dir, n_jobs=2)
mne.write_source_spaces(source_dir+ '/'+ fs_sub+'-oct6-src.fif', src_space) # write to source dir
# src_space = mne.read_source_spaces(source_dir+ '/'+ fs_sub+'-oct6-src.fif') # if you need to read in
# compute Boundary Element Model
# use os.system to run this in tcsh shell setup freesurfer and subject dir as well and make BEM seperation
os.system(f"tcsh -c 'freesurfer_6.0.0 && setenv SUBJECTS_DIR && mne watershed_bem -s {fs_sub} -d {fs_sub_dir}'")
model = mne.make_bem_model(fs_sub, subjects_dir=fs_sub_dir)  # make BEM model in MNE
# TODO: Sometimes this fails because the child's head is not very good with watershed. Fall back to one layer BEM
# model = mne.make_bem_model(fs_sub, subjects_dir=fs_sub_dir, conductivity=[0.3])
mne.write_bem_surfaces(f'{source_dir}/{fs_sub}-5120-5120-5120-bem.fif', model)  # save to source dir
bem_sol = mne.make_bem_solution(model) # make bem solution using model
mne.write_bem_solution(f'{source_dir}/{fs_sub}-5120-5120-5120-bem-sol.fif', bem_sol)
# check the BEM solution if needed
# mne.viz.plot_bem(subject=fs_sub, subjects_dir=fs_sub_dir, brain_surfaces='white', src=src_space, orientation='coronal')
# bem_sol = mne.read_bem_solution(f'{source_dir}/{fs_sub}-5120-5120-5120-bem-sol.fif') # if needed

# Co-registration # TODO: We cannot currently do this over remote ssh, make a loop for all participants
mne.gui.coregistration(inst=f'{mne_save_dir}/{num}_{f_only[2]}_epo.fif', subject=fs_sub, subjects_dir=fs_sub_dir)
trans = f'{mne_save_dir}/{num}_{f_only[2]}_coreg-trans.fif'

# compute forward solution
fwd = mne.make_forward_solution(raw.info, trans, src_space, bem_sol)
mne.write_forward_solution(f'{source_dir}/{fs_sub}_{f_only[2]}-fwd.fif', fwd) #save
# fwd = mne.read_forward_solution(f'{source_dir}/{fs_sub}_{f_only[2]}-fwd.fif') # if needed

# compute covariance matrix on epochs
cov = mne.compute_covariance(epochs, method=['shrunk', 'empirical'], rank=None) # this is quicker still takes a while
mne.write_cov(f'{source_dir}/{fs_sub}_{f_only[2]}-cov.fif', cov) # save this file
# cov = mne.read_cov(f'{source_dir}/{fs_sub}_{f_only[2]}-cov.fif') # if needed

# calculate minimum norm inverse operator
inv = mne.minimum_norm.make_inverse_operator(raw.info, fwd, cov, loose=0.2, depth=0.8)

#%% compute inverse solultion

# MNE
stc_mne = mne.minimum_norm.apply_inverse(evoked, inv, lambda2=1. / 9.)

# plot
stc_mne.plot(subjects_dir=fs_sub_dir, backend='matplotlib', hemi='lh', initial_time=.3)

# plot using peak getter at the time point and spatial point of peak
vertno_max, time_max = stc_mne.get_peak(hemi='lh')
surfer_kwargs = dict(
    hemi='lh', subjects_dir=fs_sub_dir, views='lat',
    initial_time=time_max, time_unit='s', size=(800, 800), smoothing_steps=5, backend='matplotlib')
brain = stc_mne.plot(**surfer_kwargs)
#%% look at mismatch responses

MNN = mne.combine_evoked([evokeds[0], -evokeds[1], -evokeds[2]], weights='equal')
MNN_word = mne.combine_evoked([evokeds[0], -evokeds[1]], weights='equal')
MNN_nonword = mne.combine_evoked([evokeds[0], -evokeds[2]], weights='equal')
MNN_diff = mne.combine_evoked([MNN_word, -MNN_nonword], weights='equal')


#mne_preprocess.process_multiple()
