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
                                                njobs=10)
#%% or use cluster (takes about 15-20  minutes, make yourself a coffee)
saved_list = red_preprocess.preprocess_cluster(flist=flist,
                                                indir=rawdir,
                                                outdir=rawdir,
                                               scriptpath = '/imaging/ai05/phono_oddball/fs_scripts',
                                               pythonpath = '/imaging/local/software/miniconda/envs/mne0.19/bin/python',
                                                overwrite=False)


#%% some of these mothers have got eog or ecg channels, we need to manually check these
filtered_in = [f for f in os.listdir(rawdir) if 'clean_raw' in f]

man_ica = [f for f in filtered_in if 'no' in f]
#%%
i +=1
f = man_ica[i]
raw = mne.io.read_raw_fif(f'{rawdir}/{f}', preload=True)
ica = mne.preprocessing.ICA(n_components=25, method='fastica').fit(raw)
comps = ica.plot_components()
comps[0].savefig('/home/ai05/comp.png')
comps[1].savefig('/home/ai05/comp2.png')
raw.plot(start=120).savefig('/home/ai05/raw.png')
#%% change inds and decide
ica.exclude =[1,8,9,17]
ica.apply(raw)
# if you need to plot the channels
raw.plot(start=120).savefig('/home/ai05/raw.png')
#%%
raw.save(f'{rawdir}/{f.split("_")[0]}_{f.split("_")[1]}_clean_raw.fif', overwrite=True)

#%% if you need to plot the channels
raw.plot(start=120).savefig('/home/ai05/raw.png')
#%% EPOCHED

filtered_in = [f for f in os.listdir(rawdir) if 'clean_raw' in f]

# epoch files from save list
keys = {'Freq': 10, 'Dev Word': 11, 'Dev Non-Word': 12}  # pass in keys
trigchan = 'STI101_up'  # pass in the trigger channel
backup_trigchan = 'STI102'
saved_epoch_list = red_epoch.epoch_multiple(flist=filtered_in,
                                            indir=rawdir,
                                            outdir=mne_epo_out,
                                            keys=keys,
                                            trigchan=trigchan,
                                            backup_trigchan=backup_trigchan,
                                            times=[-0.3, 1.0],
                                            overwrite=True,
                                            njobs=17)

#%% merge those epochs
epo_base = [os.path.basename(f) for f in saved_epoch_list]
epo_nums = set([f.split('_')[0] for f in epo_base])


def merge_epo(num, mne_epo_out, saved_epoch_list):

    if os.path.isfile(f'{mne_epo_out}/{num}_concat_epo.fif'):
        print(f'{mne_epo_out}/{num}_concat_epo.fif')
        print('exists, skipping')
        return

    nums_fs = [f for f in saved_epoch_list if '/'+num+'_' in f]
    merge_l = []
    for file in nums_fs:
        epo = mne.read_epochs(file)
        merge_l.append(epo)
    merged = mne.epochs.concatenate_epochs(merge_l)
    merged.save(f'{mne_epo_out}/{num}_concat_epo.fif', overwrite=True)

joblib.Parallel(n_jobs=30)(
    joblib.delayed(merge_epo)(num, mne_epo_out, saved_epoch_list) for num in epo_nums)

for num in epo_nums:
    merge_epo(num, mne_epo_out, saved_epoch_list)

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
flist = [f for f in os.listdir(mne_epo_out) if '_concat_' in f]

# run the process for getting evoked
saved_evoked_list = red_epoch.evoked_multiple(flist=flist,
                                              indir=mne_epo_out,
                                              outdir=mne_evo_out,
                                              keys=keys,
                                              contlist=contlist,
                                              contlist2=contlist2,
                                              overwrite=True,
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
fs_dir_all = os.listdir(fs_sub_dir)

# select only the scaled and coregister versions
#  use only scales or avscaled fnames
fs_scaled = [f for f in fs_dir_all if 'scaled' in f]

mne_src_files = red_sourcespace_setup.setup_src_multiple(sublist=fs_scaled,
                                                         fs_sub_dir=fs_sub_dir,
                                                         outdir=mne_src_dir,
                                                         spacing='oct6',
                                                         surface='white',
                                                         src_mode='cortical',
                                                         n_jobs1=12,
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

#%%
#srcfs = [f'{f.split("_")[0]}_{f.split("_")[1]}_white-oct6-volume-src.fif' for f in srcfs]
#%% get a forward solution for them
mne_fwd_files = red_inv.fwd_solution_multiple(megfs,
                                              transfs,
                                              srcfs,
                                              bemfs,
                                              rawdir,
                                              mne_src_dir,
                                              mne_src_dir,
                                              n_jobs=16)

#%% combine runs for each participant
#get epoched files for this
allepo = [f for f in os.listdir(mne_epo_out) if 'concat_epo.fif' in f]
eponum = set([f.split('_')[0] for f in allepo]) # parts


# add file list
allepo = [f'{mne_epo_out}/{f}' for f in allepo]
#%% compute covariance matrix


#%%
cov_files = red_inv.cov_matrix_multiple_cluster(epochlist=allepo,
                                                method='empirical',
                                                rank=None,
                                                tmax=0,
                                                outdir='/imaging/ai05/phono_oddball/mne_cov_run',
                                                pythonpath='/home/ai05/anaconda3/envs/mne_2/bin/python',
                                                scriptpath='/home/ai05/clusterscripts'
                                                )
#%%
cov_files = red_inv.cov_matrix_multiple(epochlist=allepo,
                                        method='empirical',
                                        rank=None,
                                        tmax=0,
                                        outdir='/imaging/ai05/phono_oddball/mne_cov_run',
                                        njobs=15
                                        )
#%% compute an inverse solution
# need 3 lists of files
ids = [f.split('_')[0] for f in bem_nos]

inraw, infwd, incov = [],  [], []
allrs = [f for f in os.listdir(rawdir) if os.path.isfile(rawdir+'/'+f)]
allrs = [f for f in allrs if '_1_' in f]
allss = [f for f in os.listdir(mne_src_dir) if os.path.isfile(mne_src_dir+'/'+f)]
allss = [f for f in allss if 'volume' not in f]
allcov = [f for f in os.listdir('/imaging/ai05/phono_oddball/mne_cov_run')]
# get list of raws
for id in ids:
    raws = [f for f in allrs if id in f]
    alls = [f for f in allss if id in f]
    allcs = [f for f in allcov if id in f]
    allcs = [f for f in allcs if 'concat' in f]
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
                                    njobs=15)

#%% check the files
evoked =[mne_evo_out+'/'+f for f in os.listdir(mne_evo_out) if 'MNN_ave' in f]
ev_nums = [os.path.basename(f).split('_')[0] for f in evoked]
fname_inv = [f for f in os.listdir(mne_src_dir) if 'concat-inv' in f]
fname_inv = [f'{mne_src_dir}/{f}' for f in fname_inv]
invs = [[i for i in fname_inv if n in i] for n in ev_nums] #list
invs = [f[0] if len(f) > 0 else '' for f in invs] # first or empty

for ind, i_d in enumerate(invs):
    if '' in [invs[ind]]:
        del evoked[ind]; del invs[ind];
#%%
plot_sources(evoked, invs,'/home/ai05/',fs_sub_dir );

#%% source estimates for all evoked responses -- setup files

# get all evoked files in folder
all_evo = [f for f in os.listdir(mne_evo_out) if os.path.isfile(f'{mne_evo_out}/{f}')]

# get all fsubs
all_fsub = [f for f in os.listdir(fs_sub_dir) if os.path.isdir(f'{fs_sub_dir}/{f}')]
all_fsub = [f for f in all_fsub if 'scaled' in f]

# get word MNN responses
words = [f for f in all_evo if 'MNN-Word' in f]
wordfs =[]
word_invs = []
for i in range(len(words)):
    num = words[i].split('_')[0]
    alln = [i for i in all_fsub if num in i]
    match = [i for i in alln if i.split('_')[0] == num]
    wordfs.append(match[0])

    alln = [i for i in fname_inv if num in i]
    match = [i for i in alln if os.path.basename(i).split('_')[0] == num]
    if len(match) > 0:
        word_invs.append(match[0])
    else:
        word_invs.append('')


words = [f'{mne_evo_out}/{f}' for f in words]

# get non-word MNN responses
# get word MNN responses
non_words = [f for f in all_evo if 'MNN-Non-Word' in f]
non_wordfs =[]
non_word_invs = []
for i in range(len(non_words)):
    num = non_words[i].split('_')[0]
    alln = [i for i in all_fsub if num in i]
    match = [i for i in alln if i.split('_')[0] == num]
    non_wordfs.append(match[0])

    alln = [i for i in fname_inv if num in i]
    match = [i for i in alln if os.path.basename(i).split('_')[0] == num]
    if len(match) > 0:
        non_word_invs.append(match[0])
    else:
        non_word_invs.append('')

non_words = [f'{mne_evo_out}/{f}' for f in non_words]
#%% run the first one
[[os.path.basename(f).split('_')[0], s.split('_')[0], os.path.basename(p).split('_')[0]] for f, s, p in zip(words, wordfs, word_invs)]

#%%
word_src = red_inv.invert_multiple(evokedfs=words,
                                       invfs=word_invs,
                                       lambda2 = 3,
                                       method='dSPM',
                                       morph=True,
                                       fsdir=fs_sub_dir,
                                       fssub=wordfs,
                                       outdir='/imaging/ai05/phono_oddball/mne_ev_src',
                                       njobs=18)
#%% Now the second one
non_word_src = red_inv.invert_multiple(evokedfs=non_words,
                                       invfs=non_word_invs,
                                       lambda2 = 3,
                                       method='dSPM',
                                       morph=True,
                                       fsdir=fs_sub_dir,
                                       fssub=non_wordfs,
                                       outdir='/imaging/ai05/phono_oddball/mne_ev_src',
                                       njobs=18)


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
innw = [f[0] for f in inNword]
X = red_group.src_concat_mat([inw, innw], 'fsaverage')

#%% plot an average
import numpy as np
import matplotlib.pyplot as plt

# read in dummy stc
stc = mne.read_source_estimate(inw[0][0:-7], 'fsaverage')
# replace it's data with an average
stc.data = np.average(np.abs(X[:,:,:,0]) - np.abs(X[:,:,:,1]), axis=2)
#stc.data = np.average(X[:,:,:,0], axis=2)
#stc.data = (X[:,:,1,0] - X[:,:,1,1])

#%%plot
stc.plot(backend='matplotlib', initial_time=0.4,
         smoothing_steps=5).savefig('/home/ai05/test_SPM.png')
#%% generate label then plot activity
for i in range(X.shape[2]):
    print(i)
    stc.data = (X[:,:,i,0])

    #stc = mne.read_source_estimate(inw[11][0:-7], 'fsaverage')
    #stc.data = np.average(X[:,:,:,0], axis=2)

    src_fname = '/imaging/ai05/phono_oddball/mne_source_models/fsaverage-ico5-src.fif'
    src = mne.read_source_spaces(src_fname)
    aparc_label_name = 'bankssts-lh'
    tmin, tmax = 0.0, 1.0
    stc_mean = stc.copy().crop(tmin, tmax).mean()
    label = mne.read_labels_from_annot('fsaverage', parc='aparc',
                                       subjects_dir=fs_sub_dir,
                                       regexp=aparc_label_name)[0]
    stc_mean_label = stc_mean.in_label(label)
    data = np.abs(stc_mean_label.data)
    stc_mean_label.data[data < 0.6 * np.max(data)] = 0.
    func_labels, _ = mne.stc_to_label(stc_mean_label, src=src, smooth=True,
                                      subjects_dir=fs_sub_dir, connected=True,
                                      verbose='error')
    func_label = func_labels[0]
    stc_func_label = stc.in_label(func_label)
    pca_func = stc.extract_label_time_course(func_label, src, mode='pca_flip')[0]
    pca_func *= np.sign(pca_func[np.argmax(np.abs(pca_func))])
    plt.plot(1e3 * stc_func_label.times, pca_func, 'b',
             label='Functional %s' % aparc_label_name)

plt.savefig(f'/home/ai05/time_SPM.png')
# Cool, if this looks sensible then continue with cluster perms below
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
X_con = X[:, :, :, 0] #- X[:, :, :, 1] # paired contrast



#    Note that X needs to be a multi-dimensional array of shape
#    samples (subjects) x time x space, so we permute dimensions
X_con = np.transpose(X_con, [2, 1, 0])
np.save(f'{mne_save_dir}/group_summary.npy', X_con)
# set parallel things
mne.set_memmap_min_size('1M')
mne.set_cache_dir('/tmp')
#%%
#    Now let's actually do the clustering. This can take a long time...
#    Here we set the threshold quite high to reduce computation.
p_threshold = 0.01
t_threshold = -stats.distributions.t.ppf(p_threshold / 2., 70 - 1)
#t_threshold = 1.5
print('Clustering.')
T_obs, clusters, cluster_p_values, H0 = clu = \
    spatio_temporal_cluster_1samp_test(X_con, connectivity=connectivity, n_jobs=10,
                                       threshold=t_threshold,
                                       verbose=True, n_permutations=1000, buffer_size=500)
#    Now select the clusters that are sig. at p < 0.05 (note that this value
#    is multiple-comparisons corrected).
good_cluster_inds = np.where(cluster_p_values <= 0.05)[0]

#%% submit to cluster

red_group.submit_cluster_perm(Xf = f'{mne_save_dir}/group_summary.npy',
                              srcf = src_fname,
                              jobs = 28,
                              buffer= 1000,
                              t_thresh=2.5,
                              scriptpath='/home/ai05',
                              pythonpath='/home/ai05/anaconda3/envs/mne/bin/python',
                              outpath='/imaging/ai05/phono_oddball'
                             )
#%% visualise

#    Now let's build a convenient representation of each cluster, where each
#    cluster becomes a "time point" in the SourceEstimate
stc_all_cluster_vis = summarize_clusters_stc(clu, tstep=est.tstep,
                                             vertices=fsave_vertices,
                                             subject='fsaverage')


#%% rename some bits
allevo = os.listdir(mne_evo_out)
for f in allevo:
    nf = mne_evo_out + '/' + f
    new = nf.replace('_raw.fif_', '_concat_')
    os.rename(nf, new)
