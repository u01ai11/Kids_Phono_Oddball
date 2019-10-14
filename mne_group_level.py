#%%
import RedMegTools.epoch as red_epoch
import RedMegTools.preprocess as red_preprocess
import RedMegTools.sourcespace as red_sourcespace
import os
import collections

# pointers to our directories
rawdir = '/imaging/ai05/phono_oddball/maxfilt_raws'  # raw fifs to input
mne_save_dir = '/imaging/ai05/phono_oddball/mne_files'  # where to save MNE MEG files
source_dir = '/imaging/ai05/phono_oddball/mne_source_models'  # where to save MNE source recon files
struct_dir = '/imaging/ai05/phono_oddball/structurals_renamed'  # where our structs are
fs_sub_dir = '/imaging/ai05/phono_oddball/fs_subdir'  # fresurfer subject dir

#%% PREPROCESSING

# set to use only a few processers on this node
os.sched_setaffinity(0, {0,1,2,3,4,5,6,7,8})

# make a list of files for pre-processing
flist = [f for f in os.listdir(rawdir) if os.path.isfile(os.path.join(rawdir, f))]

# preprocess those files
saved_list = red_preprocess.preprocess_multiple(flist=flist,
                                                indir=rawdir,
                                                outdir=mne_save_dir,
                                                overwrite=False,
                                                njobs=8)

#%% EPOCHED
# make a list of files for epoching from above
flist = [os.path.basename(x) for x in saved_list]  # get filenames only

# epoch files from save list
keys = {'Freq': 10, 'Dev Word': 11, 'Dev Non-Word': 12}  # pass in keys
trigchan = 'STI101_up'  # pass in the trigger channel
backup_trigchan = 'STI102'
saved_epoch_list = red_epoch.epoch_multiple(flist=flist,
                                            indir=mne_save_dir,
                                            outdir=mne_save_dir,
                                            keys=keys,
                                            trigchan=trigchan,
                                            backup_trigchan=backup_trigchan,
                                            times=[-0.2, 0.8],
                                            overwrite=False,
                                            njobs=1)

#%% EVOKED
# compute evoked files from epoch list
# list of contrasts in ordered Dict -- first level contrasts
# These indices refer to the keys above
# The sign refers to their weighting
contlist = collections.OrderedDict({
    'MNN': [0, -1, -2],
    'MNN Word': [0, -1],
    'MNN Non-Word': [0, -2]
})

# list of second level contrasts (i.e. combining the above simple contrasts)
# indiced match onto first order contrasts
contlist2 = collections.OrderedDict({
    'Word-Nonword': [1, -2]
})

# get an input file name list
flist = [os.path.basename(x) for x in saved_epoch_list]

# run the process for getting evoked
saved_evoked_list = red_epoch.evoked_multiple(flist=flist,
                                              indir=mne_save_dir,
                                              outdir=mne_save_dir,
                                              keys=keys,
                                              contlist=contlist,
                                              contlist2=contlist2,
                                              overwrite=False,
                                              njobs=8)

#%% FREESURFER RECON
subnames_only = list(set([x.split('_')[0] for x in flist])) # get a unique list of IDs

fs_recon_list = red_sourcespace.recon_all_multiple(sublist=subnames_only,
                                                   struct_dir= struct_dir,
                                                   fs_sub_dir=fs_sub_dir,
                                                   fs_script_dir='/imaging/ai05/phono_oddball/fs_scripts',
                                                   fs_call='freesurfer_6.0.0',
                                                   njobs=1,
                                                   cbu_clust=True,
                                                   cshrc_path='/home/ai05/.cshrc')

#%% make BEM model
# get all participants who have a fs_dir
fs_dir_all = os.listdir(fs_sub_dir)
fs_dir_subs = [f for f in fs_dir_all if f in subnames_only]
# run run
fs_recon_list = red_sourcespace.fs_bem_multiple(sublist=fs_dir_subs,
                                                fs_sub_dir=fs_sub_dir,
                                                fs_script_dir='/imaging/ai05/phono_oddball/fs_scripts',
                                                fs_call='freesurfer_6.0.0',
                                                njobs=1,
                                                cbu_clust=True,
                                                cshrc_path='/home/ai05/.cshrc')
