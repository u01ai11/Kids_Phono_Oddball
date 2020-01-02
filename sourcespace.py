import sys
sys.path.insert(0, '/home/ai05/Kids_Phono_Oddball')
import RedMegTools.group as red_group
import os
import mne
import numpy as np
import copy

# pointers to our directories
rawdir = '/imaging/ai05/phono_oddball/aligned_raws'  # raw fifs to input
mne_save_dir = '/imaging/ai05/phono_oddball/mne_files'  # where to save MNE MEG files
source_dir = '/imaging/ai05/phono_oddball/mne_source_models'  # where to save MNE source recon files
struct_dir = '/imaging/ai05/phono_oddball/structurals_renamed'  # where our structs are
fs_sub_dir = '/imaging/ai05/phono_oddball/fs_subdir'  # fresurfer subject dir
mne_src_dir = '/imaging/ai05/phono_oddball/mne_source_models'
mne_epo_out = '/imaging/ai05/phono_oddball/mne_epoch'
mne_evo_out = '/imaging/ai05/phono_oddball/mne_evoked_sess'
src_ev_dir = '/imaging/ai05/phono_oddball/mne_ev_src'
inv_dir = '/imaging/ai05/phono_oddball/mne_inv_parts'
flist = [f for f in os.listdir(rawdir) if os.path.isfile(os.path.join(rawdir, f))]
subnames_only = list(set([x.split('_')[0] for x in flist])) # get a unique list of IDs
os.chdir('/imaging/ai05/phono_oddball/cluster_logs')

#%% This can get a bit spicy in terms of resources, so restrict CPUs we can use
# set to use only a few processers on this node
os.sched_setaffinity(0, {0,1,2,3,4,5,6,7,8})

#also restrict memory
# import resource
# soft, hard = resource.getrlimit(resource.RLIMIT_AS)
# #80gb memory limit
# resource.setrlimit(resource.RLIMIT_AS, (1000*1000*10000, hard))

#%% get basename for stc files from directory for words and non-words
src_ev_dir = '/imaging/ai05/phono_oddball/mne_ev_src'
w_lh = [f for f in os.listdir(src_ev_dir) if 'MNN-Word-lh' in f]
w_rh = [f for f in os.listdir(src_ev_dir) if 'MNN-Word-rh' in f]
n_lh = [f for f in os.listdir(src_ev_dir) if 'Non-Word-lh' in f]
n_rh = [f for f in os.listdir(src_ev_dir) if 'Non-Word-rh' in f]

#%% read thes in and create big matrix
# inw = [f[0] for f in inWord]
# innw = [f[0] for f in inNword]
ws = [f.split('-lh')[0] for f in w_lh]
ws = [f'{src_ev_dir}/{f}' for f in ws]
ns = [f.split('-lh')[0] for f in n_lh]
ns = [f'{src_ev_dir}/{f}' for f in ns]

# match these lists -- as there are some recons that failed for certain evoked
# .. objects but not others
wbase= [os.path.basename(f).split('MNN')[0] for f in ws]
nbase = [os.path.basename(f).split('MNN')[0] for f in ws]

#loop through word_base
fws = []
fns = []
for i in range(len(wbase)):
    match_n = [f for f in nbase if wbase[i] in f]
    if len(match_n) > 0:
        fws.append(src_ev_dir+'/'+match_n[0]+'MNN-Word')
        fns.append(src_ev_dir + '/' + match_n[0] + 'MNN-Non-Word')
    else:
        print('not matched for '+ match_n[0])

#%% read in
X = red_group.src_concat_mat([fws,fns], 'fsaverage')

#%% take difference
og_X = copy.deepcopy(X)

X = og_X[:,:,:,1]
#%% filter out participants with non-spatial specific responses
# This has happened on some subjects due to annoying gradioemeter noise issues
# but also because some children where so far away from the sensors that we ran
# into serious issues

# take SD along the time axis for each participant
stds = []
for i in range(X.shape[2]):
    meanmap = np.mean(X[:, :, i], axis = 1)
    stds.append(meanmap.std())
    perc_done = i / X.shape[2]
    sys.stdout.write("\rReading %i percent" % round(perc_done * 100, 2))
    sys.stdout.flush()
#%% get the silly indices of those poor recon files as a mask
silly_mask = np.array(stds) < 10000
X_filt = X[:,:,silly_mask]
#%% plot an average
# read in dummy stc
stc = mne.read_source_estimate(ws[0], 'fsaverage')
# make an average
xav = np.average(X_filt[:,:,:], axis=2)
# replace in the dummy object
stc.data = xav

#%%
stc.plot(backend='matplotlib', initial_time=0.75,
         hemi='lh').savefig('/imaging/ai05/images/nonword_SPM.png')
#%%plot to investigate sensible-ness
for i in range(0,1050, 50):
    stc.plot(backend='matplotlib', initial_time=i/1000,
             hemi='lh').savefig('/imaging/ai05/images/word_SPM.png')

#%% now that is done let's run some stats - start with simple cluster t test

#first get a connectivity matrix. This is a matrix that defines which voxels are
# connected
src_fname = '/imaging/ai05/phono_oddball/mne_source_models/fsaverage-ico5-src.fif'
src = mne.read_source_spaces(src_fname)
fsave_vertices = [s['vertno'] for s in src]
connectivity = mne.spatial_src_connectivity(src)

#%% one sample test

inX = np.transpose(X[0], [2, 1, 0]) # transpose for MNE sub x time x space
#%%
threshold = 0.2  # t thresh
# set family-wise p-value
p_accept = 0.05

T_obs, clusters, cluster_p_values, H0 = clu = \
    mne.stats.spatio_temporal_cluster_1samp_test(np.transpose(X[:,:,:,0]-X[:,:,:,1], [2, 1, 0]), connectivity=connectivity, n_jobs=3,
                                       threshold=threshold, buffer_size=None,
                                       verbose=True, n_permutations=100)


#%% Do we have any good clusters?
good_cluster_inds = np.where(cluster_p_values < 0.05)[0]

#%% two sample test
# next prepare data for MNE, it expects certain dimensopns
inX = og_X
inX = inX[:,:,silly_mask,:] # filter out stupid noisy recons
inX = np.transpose(inX, [2, 1, 0, 3])

#%% run
threshold = 3.0  # very high, but the test is quite sensitive on this data
# set family-wise p-value
p_accept = 0.05

inlist = [inX[:,:,:,0], inX[:,:,:,1]]

cluster_stats = mne.stats.spatio_temporal_cluster_test([np.transpose(X[:,:,:,0], [2, 1, 0])], n_permutations=10,
                                             threshold=threshold, tail=1,
                                             n_jobs=1, buffer_size=None,
                                             connectivity=connectivity)
#%% Do we have any good clusters?
T_obs, clusters, p_values, _ = cluster_stats
good_cluster_inds = np.where(p_values < p_accept)[0]


