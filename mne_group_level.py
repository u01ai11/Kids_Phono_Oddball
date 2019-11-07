#%%
import sys
sys.path.insert(0, '/home/ai05/Kids_Phono_Oddball')
import RedMegTools.epoch as red_epoch
import RedMegTools.preprocess as red_preprocess
import RedMegTools.sourcespace_command_line as red_sourcespace_cmd
import RedMegTools.sourcespace_setup as red_sourcespace_setup
import RedMegTools.utils as red_utils
import RedMegTools.inversion as red_inv
import RedMegTools.group as red_group
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
src_ev_dir = '/imaging/ai05/phono_oddball/mne_ev_src'
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
                                                outdir='/imaging/ai05/phono_oddball/mne_cov_run',
                                                pythonpath='/home/ai05/anaconda3/envs/mne/bin/python',
                                                scriptpath='/home/ai05/clusterscripts'
                                                )
#%%
cov_files = red_inv.cov_matrix_multiple(epochlist=allepo,
                                        method='empirical',
                                        rank=None,
                                        tmax=0,
                                        outdir='/imaging/ai05/phono_oddball/mne_cov_run',
                                        njobs=10
                                        )
#%% compute an inverse solution
# need 3 lists of files
ids = [f.split('_')[0] for f in bem_nos]

inraw, infwd, incov = [],  [], []
allrs = [f for f in os.listdir(rawdir) if os.path.isfile(rawdir+'/'+f)]
allss = [f for f in os.listdir(mne_src_dir) if os.path.isfile(mne_src_dir+'/'+f)]
allcov = [f for f in os.listdir('/imaging/ai05/phono_oddball/mne_cov_run')]
# get list of raws
for id in ids:
    raws = [f for f in allrs if id in f]
    raws = [f for f in raws if 'concat' in f]
    alls = [f for f in allss if id in f]
    allcs = [f for f in allcov if id in f]
    if len(raws) > 0:
        inraw.append(rawdir + '/' + raws[0])
    else:
        inraw.append('')

    fwds = [f for f in alls if 'fwd.fif' in f]
    if len(fwds) > 0:
        infwd.append(mne_src_dir + '/' + fwds[0])
    else:
        infwd.append('')

    covs = [f for f in allcs if '-cov.fif' in f]
    if len(covs) > 0:
        incov.append('/imaging/ai05/phono_oddball/mne_cov_run' + '/' + covs[0])
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


for i in range(len(incov)):
    cov = mne.read_cov(incov[i])
    if numpy.sum(cov.data) == 0:
        print(f'sucky {incov[i]}')


#%% compute an inverse solution
fname_inv = red_inv.inv_op_multiple(infofs=inraw,
                                    fwdfs=infwd,
                                    covfs=incov,
                                    loose=0.2,
                                    depth=0.8,
                                    outdir=mne_src_dir,
                                    njobs=1)

#%% check the files
evoked =[mne_evo_out+'/'+f for f in os.listdir(mne_evo_out) if 'concat_ave' in f]
ev_nums = [f.split('_')[0] for f in evoked]
invs = [[i for i in fname_inv if n in i] for n in ev_nums] #list
invs = [f[0] if len(f) > 0 else '' for f in invs] # first or empty

for ind, i_d in enumerate(invs):
    if '' in [invs[ind]]:
        del evoked[ind]; del invs[ind];
#%%
plot_sources(evoked, invs,'/home/ai05/',fs_sub_dir )

#%% source estimates for all evoked responses -- setup files

# get all evoked files in folder
all_evo = [f for f in os.listdir(mne_evo_out) if os.path.isfile(f'{mne_evo_out}/{f}')]

# get all fsubs
all_fsub = [f for f in os.listdir(fs_sub_dir) if os.path.isdir(f'{fs_sub_dir}/{f}')]
all_fsub = [f for f in all_fsub if 'scaled' in f]

# get word MNN responses
words = [f for f in all_evo if 'MNN-Word' in f]
wordfs =[]
for i in range(len(words)):
    num = words[i].split('_')[0]
    for ii in range(len(all_fsub)):
        if num in all_fsub[ii]:
            wordfs.append(all_fsub[ii])
words_invs = [[i for i in fname_inv if n.split('_')[0] in i] for n in words]  # list
words_invs = [i[0] if i != [] else '' for i in words_invs]
words = [f'{mne_evo_out}/{f}' for f in words]

# get non-word MNN responses
non_words = [f for f in all_evo if 'MNN-Non-Word' in f]
non_wordfs = []
for i in range(len(non_words)):
    num = non_words[i].split('_')[0]
    for ii in range(len(all_fsub)):
        if num in all_fsub[ii]:
            non_wordfs.append(all_fsub[ii])
non_words_invs = [[i for i in fname_inv if n.split('_')[0] in i] for n in non_words] #list
non_words_invs = [i[0] if i != [] else '' for i in non_words_invs]
non_words = [f'{mne_evo_out}/{f}' for f in non_words]

#%% run the first one

non_word_src = red_inv.invert_multiple(evokedfs=words,
                                       invfs=words_invs,
                                       lambda2 = 1. / 9.,
                                       method='dSPM',
                                       morph=True,
                                       fsdir=fs_sub_dir,
                                       fssub=wordfs,
                                       outdir='/imaging/ai05/phono_oddball/mne_ev_src',
                                       njobs=1)
#%% Now the second one
non_word_src = red_inv.invert_multiple(evokedfs=non_words,
                                       invfs=non_words_invs,
                                       lambda2 = 1. / 9.,
                                       method='dSPM',
                                       morph=True,
                                       fsdir=fs_sub_dir,
                                       fssub=non_wordfs,
                                       outdir='/imaging/ai05/phono_oddball/mne_ev_src',
                                       njobs=1)


#%% get average of each for just looking at
src_ev_dir = '/imaging/ai05/phono_oddball/mne_ev_src'

src_evs = [f for f in os.listdir(src_ev_dir) if os.path.isfile(f'{src_ev_dir}/{f}')]
wrd_ids = list(set([f.split('_')[0] for f in src_evs]))
inWord = []
inNword = []
for ID in wrd_ids:
    thisids = [f for f in src_evs if ID in f]
    thisword = [f'{src_ev_dir}/{f}' for f in thisids if 'MNN-Word' in f]
    thisword = [[f for f in thisword if 'rh' in f][0], [f for f in thisword if 'lh' in f][0]]
    inWord.append(thisword)
    thisnonword = [f'{src_ev_dir}/{f}' for f in thisids if 'Non-Word' in f]
    thisnonword = [[f for f in thisnonword if 'rh' in f][0], [f for f in thisnonword if 'lh' in f][0]]
    inNword.append(thisnonword)

#%% look at a couple of plots
est = mne.read_source_estimate(inWord[20][0], 'fsaverage')
#est = mne.read_source_estimate(src_ev_dir+'/'+src_evs[2], 'fsaverage')
vertno_max, time_max = est.get_peak()
est.plot(backend='matplotlib', initial_time=time_max,
         smoothing_steps=5).savefig('/home/ai05/test.png')

#%% read and create big matrix
inw = [f[0] for f in inWord]
innw = [f[0] for f in inWord]
X = red_group.src_concat_mat([inw, innw], 'fsaverage')
#%% do manually
import mne
import numpy as np

avlist_w = []
avlist_n = []

for pair in inWord:
    est = mne.read_source_estimate(pair[0][0:-7], 'fsaverage')
    avlist_w.append(est)

for pair in inNword:
    est = mne.read_source_estimate(pair[0][0:-7], 'fsaverage')
    avlist_n.append(est)

X_w = np.empty((avlist_w[0].shape[0], avlist_w[0].shape[1], len(avlist_w)))
X_n = np.empty((avlist_n[0].shape[0], avlist_n[0].shape[1], len(avlist_n)))

for i in range(len(avlist_w)):
    X_w[:,:,i] = avlist_w[i].data

for i in range(len(avlist_n)):
    X_n[:,:,i] = avlist_n[i].data

X_av = np.average(X_n, axis=2)

# est.data = X_av
#
# vertno_max, time_max = est.get_peak()
# est.plot(backend='matplotlib', initial_time=time_max,
#          smoothing_steps=5, hemi='rh').savefig('/home/ai05/test.png')

X = np.empty((avlist_n[0].shape[0], avlist_n[0].shape[1], len(avlist_n), 2))
X[:,:,:,0] = X_w
X[:,:,:,1] = X_n
#%% cluster perm
from scipy import stats as stats
from mne.stats import (spatio_temporal_cluster_1samp_test,
                       summarize_clusters_stc)
src_fname = '/imaging/ai05/phono_oddball/mne_source_models/fsaverage-ico5-src.fif'
src = mne.read_source_spaces(src_fname)
fsave_vertices = [s['vertno'] for s in src]
connectivity = mne.spatial_src_connectivity(src)

X = np.abs(X)
X_con = X[:, :, :, 0] - X[:, :, :, 1] # paired contrast

#    Note that X needs to be a multi-dimensional array of shape
#    samples (subjects) x time x space, so we permute dimensions
X_con = np.transpose(X_con, [2, 1, 0])

# set parallel things
mne.set_memmap_min_size('1M')
mne.set_cache_dir('/tmp')

#    Now let's actually do the clustering. This can take a long time...
#    Here we set the threshold quite high to reduce computation.
p_threshold = 0.001
#t_threshold = -stats.distributions.t.ppf(p_threshold / 2., 70 - 1)
t_threshold = 2
print('Clustering.')
T_obs, clusters, cluster_p_values, H0 = clu = \
    spatio_temporal_cluster_1samp_test(X_con, connectivity=connectivity, n_jobs=10,
                                       threshold=t_threshold, buffer_size=500,
                                       verbose=True)
#    Now select the clusters that are sig. at p < 0.05 (note that this value
#    is multiple-comparisons corrected).
good_cluster_inds = np.where(cluster_p_values < 0.05)[0]