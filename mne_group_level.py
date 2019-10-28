#%%
import sys
sys.path.insert(0, '/home/ai05/Kids_Phono_Oddball')
import RedMegTools.epoch as red_epoch
import RedMegTools.preprocess as red_preprocess
import RedMegTools.sourcespace_command_line as red_sourcespace_cmd
import RedMegTools.sourcespace_setup as red_sourcespace_setup
import RedMegTools.utils as red_utils
import RedMegTools.inversion as red_inv
import os
import collections

# pointers to our directories
rawdir = '/imaging/ai05/phono_oddball/maxfilt_raws'  # raw fifs to input
mne_save_dir = '/imaging/ai05/phono_oddball/mne_files'  # where to save MNE MEG files
source_dir = '/imaging/ai05/phono_oddball/mne_source_models'  # where to save MNE source recon files
struct_dir = '/imaging/ai05/phono_oddball/structurals_renamed'  # where our structs are
fs_sub_dir = '/imaging/ai05/phono_oddball/fs_subdir'  # fresurfer subject dir
mne_src_dir = '/imaging/ai05/phono_oddball/mne_source_models'
flist = [f for f in os.listdir(rawdir) if os.path.isfile(os.path.join(rawdir, f))]
subnames_only = list(set([x.split('_')[0] for x in flist])) # get a unique list of IDs

#%% PREPROCESSING

# set to use only a few processers on this node
os.sched_setaffinity(0, {0,1,2,3,4,5,6,7,8})

# make a list of files for pre-processing
flist = [f for f in os.listdir(rawdir) if os.path.isfile(os.path.join(rawdir, f))]
#%%
red_utils.align_runs_max(rawdir, 'maxfilter_2.2.12', '/imaging/ai05/phono_oddball/aligned_raws', '/imaging/ai05/phono_oddball/fs_scripts')

#%%
# preprocess those files
saved_list = red_preprocess.preprocess_multiple(flist=flist,
                                                indir=rawdir,
                                                outdir=mne_save_dir,
                                                overwrite=False,
                                                njobs=1)

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
                                            times=[-0.3, 0.8],
                                            overwrite=True,
                                            njobs=32)

#%% EVOKED
# compute evoked files from epoch list
# list of contrasts in ordered Dict -- first level contrasts
# These indices refer to the keys above
# The sign refers to their weighting
contlist = collections.OrderedDict({
    'MNN': [0, -1, -2],
    'MNN-Word': [0, -1],
    'MNN-Non-Word': [0, -2]
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
                                              njobs=32)

#%% FREESURFER RECON
subnames_only = list(set([x.split('_')[0] for x in flist])) # get a unique list of IDs

fs_recon_list = red_sourcespace_cmd.recon_all_multiple(sublist=subnames_only,
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

#use only scales or avscaled fnames
fs_dir_subs = [f for f in fs_dir_all if f not in subnames_only]

# run run run run
fs_recon_list = red_sourcespace_cmd.fs_bem_multiple(sublist=fs_dir_subs,
                                                fs_sub_dir=fs_sub_dir,
                                                fs_script_dir='/imaging/ai05/phono_oddball/fs_scripts',
                                                fs_call='freesurfer_6.0.0',
                                                njobs=1,
                                                cbu_clust=True,
                                                cshrc_path='/home/ai05/.cshrc')

#%% setup sourcespace in MNE format
mne_src_dir = '/imaging/ai05/phono_oddball/mne_source_models'

# select only the scaled and coregister versions
#  use only scales or avscaled fnames
fs_scaled = [f for f in fs_dir_all if 'scaled' in f]

mne_src_files = red_sourcespace_setup.setup_src_multiple(sublist=fs_scaled,
                                                         fs_sub_dir=fs_sub_dir,
                                                         outdir=mne_src_dir,
                                                         spacing='oct6',
                                                         surface='white',
                                                         n_jobs1=19,
                                                         n_jobs2=1)
#%% BEM MNE input stuff
mne_bem_files = red_sourcespace_setup.make_bem_multiple(sublist=fs_scaled,
                                                        fs_sub_dir=fs_sub_dir,
                                                        outdir=mne_src_dir,
                                                        single_layers=True,
                                                        n_jobs1=20)

#%% choose participants with existing bem models
# this gives us lists of files for each participant
checklist = red_utils.check_ids(rawdir, fs_sub_dir, mne_src_dir)
# this gets just there filenames and ids
bem_fs = [i for i in checklist[3] if i != '']  # strip out the empties
bem_nos = [i[0].split('-')[0] for i in bem_fs]  # get their ids

# now we need to find the corresponding raw files, trans, sourcespace and bemsols
megfs, transfs, srcfs, bemfs = red_utils.find_fwd_files(bem_nos, mne_src_dir, rawdir)

# exclude any participants with an empty
ind = 0
for mgf, trf, srf, bmf, in zip(megfs, transfs, srcfs, bemfs):
    if any(item == '' for item in  [mgf, trf, srf, bmf]):
        del megfs[ind], transfs[ind], srcfs[ind], bemfs[ind]
    ind = ind+1

#%% get a forward solution for them
mne_fwd_files = red_inv.fwd_solution_multiple(megfs, transfs, srcfs, bemfs, rawdir, mne_src_dir, mne_src_dir, n_jobs=16)

#%% combine runs for each participant
#get epoched files for this
allepo = [f for f in os.listdir(mne_save_dir) if '_epo.fif' in f]
eponum = set([f.split('_')[0] for f in allepo]) # parts


# add file list
allepo = [f'{mne_save_dir}/{f}' for f in allepo]
#%% compute covariance matrix


#%%
cov_files = red_inv.cov_matrix_multiple_cluster(epochlist=allepo,
                                                method='empirical',
                                                rank=None,
                                                tmax=0,
                                                outdir=mne_src_dir,
                                                pythonpath='/home/ai05/anaconda3/envs/mne/bin/python',
                                                scriptpath='/home/ai05/clusterscripts'
                                                )
#%%
cov_files = red_inv.cov_matrix_multiple(epochlist=allepo,
                                        method='empirical',
                                        rank=None,
                                        tmax=0,
                                        outdir=mne_src_dir,
                                        njobs=16
                                        )
#%%

