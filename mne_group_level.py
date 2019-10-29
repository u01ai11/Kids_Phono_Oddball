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
import mne
import joblib

# pointers to our directories
rawdir = '/imaging/ai05/phono_oddball/aligned_raws'  # raw fifs to input
mne_save_dir = '/imaging/ai05/phono_oddball/mne_files'  # where to save MNE MEG files
source_dir = '/imaging/ai05/phono_oddball/mne_source_models'  # where to save MNE source recon files
struct_dir = '/imaging/ai05/phono_oddball/structurals_renamed'  # where our structs are
fs_sub_dir = '/imaging/ai05/phono_oddball/fs_subdir'  # fresurfer subject dir
mne_src_dir = '/imaging/ai05/phono_oddball/mne_source_models'
mne_epo_out = '/imaging/ai05/phono_oddball/mne_epoch'
mne_evo_out = '/imaging/ai05/phono_oddball/mne_evoked'
flist = [f for f in os.listdir(rawdir) if os.path.isfile(os.path.join(rawdir, f))]
subnames_only = list(set([x.split('_')[0] for x in flist])) # get a unique list of IDs

#%% PREPROCESSING

# set to use only a few processers on this node
os.sched_setaffinity(0, {0,1,2,3,4,5,6,7,8})

# make a list of files for pre-processing
flist = [f for f in os.listdir(rawdir) if os.path.isfile(os.path.join(rawdir, f))]
#%% align runs
red_utils.align_runs_max(rawdir, 'maxfilter_2.2.12', '/imaging/ai05/phono_oddball/aligned_raws', '/imaging/ai05/phono_oddball/fs_scripts')


#%%
# preprocess those files
saved_list = red_preprocess.preprocess_multiple(flist=flist,
                                                indir=rawdir,
                                                outdir=mne_save_dir,
                                                overwrite=False,
                                                njobs=1)

#%% merge runs raws


#%% EPOCHED
# flist for combined
merge_raw = [f for f in flist if '_concat_' in f]
# make a list of files for epoching from above
#flist = [os.path.basename(x) for x in saved_list]  # get filenames only

# epoch files from save list
keys = {'Freq': 10, 'Dev Word': 11, 'Dev Non-Word': 12}  # pass in keys
trigchan = 'STI101_up'  # pass in the trigger channel
backup_trigchan = 'STI102'
saved_epoch_list = red_epoch.epoch_multiple(flist=merge_raw,
                                            indir=rawdir,
                                            outdir=mne_epo_out,
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
                                              indir=mne_epo_out,
                                              outdir=mne_evo_out,
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

#%% mop up and BEM any missing (usually from fsaverage)

indices = [i for i, x in enumerate(checklist[3]) if x == '']
mopups = [checklist[0][f][0] for f in indices if checklist[0][f] != '']

joblib.Parallel(n_jobs=len(mopups))(
           joblib.delayed(bem_mopup)(id_, mne_src_dir, fs_sub_dir) for id_ in mopups)



#%% get a forward solution for them
mne_fwd_files = red_inv.fwd_solution_multiple(megfs, transfs, srcfs, bemfs, rawdir, mne_src_dir, mne_src_dir, n_jobs=16)

#%% combine runs for each participant
#get epoched files for this
allepo = [f for f in os.listdir(mne_epo_out) if '_epo.fif' in f]
eponum = set([f.split('_')[0] for f in allepo]) # parts


# add file list
allepo = [f'{mne_epo_out}/{f}' for f in allepo]
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
                                        njobs=5
                                        )
#%% compute an inverse solution
# need 3 lists of files
ids = [f.split('_')[0] for f in bem_nos]

inraw, infwd, incov = [],  [], []
allrs = [f for f in os.listdir(rawdir) if os.path.isfile(rawdir+'/'+f)]
allss = [f for f in os.listdir(mne_src_dir) if os.path.isfile(mne_src_dir+'/'+f)]

# get list of raws
for id in ids:
    raws = [f for f in allrs if id in f]
    raws = [f for f in raws if 'concat' in f]
    alls = [f for f in allss if id in f]
    if len(raws) > 0:
        inraw.append(rawdir + '/' + raws[0])
    else:
        inraw.append('')

    fwds = [f for f in alls if 'fwd.fif' in f]
    if len(fwds) > 0:
        infwd.append(mne_src_dir + '/' + fwds[0])
    else:
        infwd.append('')

    covs = [f for f in alls if 'concat-cov.fif' in f]
    if len(covs) > 0:
        incov.append(mne_src_dir + '/' + covs[0])
    else:
        incov.append('')

# only include participant numbers with all files
for ind, i_d in enumerate(ids):
    if '' in [inraw[ind], infwd[ind], incov[ind]]:
        del inraw[ind]; del infwd[ind]; del incov[ind]; del ids[ind]

#%% check covariances
# NOTE: for some reason the parallel script creates some empty covariance matrices
# these can be fixed by recalculation (which is what the below does)

import mne, numpy
for i in range(len(incov)):
    cov = mne.read_cov(incov[i])
    if numpy.sum(cov.data) < 1:
        print(f're-calculating {incov[i]}')
        num = os.path.basename(incov[i]).split('_')[0]
        epo = f'{mne_epo_out}/{num}_concat_epo.fif'
        epochs = mne.read_epochs(epo)
        newcov = mne.compute_covariance(epochs, method='empirical', tmax=0)
        os.system(f'rm {incov[i]}')
        mne.write_cov(incov[i], newcov)



#%% compute an inverse solution
fname_inv = red_inv.inv_op_multiple(infofs=inraw,
                                    fwdfs=infwd,
                                    covfs=incov,
                                    loose=0.2,
                                    depth=0.8,
                                    outdir=mne_src_dir,
                                    njobs=32)

