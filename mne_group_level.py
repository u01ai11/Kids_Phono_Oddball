import red_meg_tools
import os

# pointers to our directories
rawdir = '/imaging/ai05/phono_oddball/maxfilt_raws'   # raw fifs to input
mne_save_dir = '/imaging/ai05/phono_oddball/mne_files'  # where to save MNE MEG files
source_dir = '/imaging/ai05/phono_oddball/mne_source_models'  # where to save MNE source recon files
struct_dir = '/imaging/ai05/phono_oddball/structurals_renamed'  # where our structs are
fs_sub_dir = '/imaging/ai05/phono_oddball/fs_subdir'  # fresurfer subject dir


# make a list of files for pre-processing
flist = [f for f in os.listdir(rawdir) if os.path.isfile(os.path.join(rawdir, f))]

# preprocess those files
saved_list = red_meg_tools.preprocess_multiple(flist=flist[0:8],
                                               indir=rawdir,
                                               outdir=mne_save_dir,
                                               overwrite=False,
                                               njobs=8)


# make a list of files for epoching from above
flist = [os.path.basename(x) for x in saved_list] # get filenames only

# epoch files from save list
keys = {'Freq': 10, 'Dev Word': 11, 'Dev Non-Word': 12}  # pass in keys
trigchan = 'STI101_up'  # pass in the trigger channel
saved_epoch_list = red_meg_tools.epoch_multiple(flist=flist[0:8],
                                                indir=mne_save_dir,
                                                outdir=mne_save_dir,
                                                keys=keys,
                                                trigchan=trigchan,
                                                times=[-0.2, 0.8],
                                                overwrite=False,
                                                njobs=8)

# compute evoked files from epoch list

