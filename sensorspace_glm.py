import sys
sys.path.insert(0, '/home/ai05/Kids_Phono_Oddball')
import RedMegTools.epoch as red_epoch
import RedMegTools.preprocess as red_preprocess
import RedMegTools.sourcespace_command_line as red_sourcespace_cmd
import RedMegTools.sourcespace_setup as red_sourcespace_setup
import RedMegTools.utils as red_utils
import RedMegTools.inversion as red_inv
import RedMegTools.group as red_group
import RedMegTools.permutations as red_perm
sys.path.insert(0, '/home/ai05/Downloads/glm')
import glmtools
import os
import collections
import mne
import joblib
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd
import scipy
from collections import namedtuple
import copy

os.chdir('/imaging/ai05/phono_oddball/cluster_logs')
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

#%% source estimates for all evoked responses -- setup files
# get all evoked files in folder
all_evo = [f for f in os.listdir(mne_evo_out) if os.path.isfile(f'{mne_evo_out}/{f}')]
# make sure they are concat files
all_evo = [f for f in all_evo if 'concat' in f]

# get numbers of participants
nums = set([f.split('_')[0] for f in all_evo])

# get behavioural factors›
words = [f'{f}_concat_MNN-Word_ave.fif' for f in nums] # file names for words
words = [f for f in words if os.path.isfile(mne_evo_out+'/'+f)]
non_words = [f'{f}_concat_MNN-Non-Word_ave.fif' for f in nums] # file names for non_words
non_words = [f for f in non_words if os.path.isfile(mne_evo_out+'/'+f)]
#check matches
nonmatches = [w.split('_')[0] != n.split('_')[0] for (w, n) in zip(words, non_words)]
print('are there any non-matching entries:')
print(any(nonmatches))

#%% add on filepaths
words = [f'{mne_evo_out}/{f}' for f in words]
non_words = [f'{mne_evo_out}/{f}' for f in non_words]
words_e = []
for ef in words:
    ev = mne.read_evokeds(ef)
    ev[0].pick('mag')
    words_e.append(ev[0])

non_words_e = []
for ef in non_words:
    ev = mne.read_evokeds(ef)
    ev[0].pick('mag')
    non_words_e.append(ev[0])


#%% Get data into acceptable format for cluster permutation testing
# we need a list of comparisons
# each item contains a 3D matrix (participant x timepoints x sensors)

X = []
for epolist in [words_e, non_words_e]:
    # append empty array
    tX = np.empty((len(epolist), len(epolist[0].times), epolist[0].data.shape[0]))
    # first dimension
    for i in range(len(epolist)):
        tX[i, :, :] = np.transpose(epolist[i].data, (1,0))
    X.append(tX)

#%% Ite let's try this using Quinn's glmtools library
import glmtools
"""
- Create a data matrix containing each participants MMN [ participants x time x voxel ]
- Add behavioural score to the data.info dict
- Create a design with two regressors
    first is a constant (mean-term)
    second is a parametric regressor containing the z-transformed behavioural score.
    
The fitted GLM will return a [2 x time x voxel] matrix in which [0,:,:] contains the mean MMN 
and [1,:,:,] contains the covariance between behavioural score and the MMN across participants.

"""

# select data -- MNN or difference
X_diff = X[0]
# as we are doing a behaviour covariate regression exclude data we have no covariate for
facts = pd.read_csv('/imaging/ai05/phono_oddball/ref_data/components_MNE.csv', header=0, index_col=0)

names_to_regress = ['WM_exec', 'Classic_IQ', 'verbalstm_wm', 'education']

regmat = np.transpose(facts[names_to_regress].to_numpy())


X_in = np.empty((len([f for f in regmat[0] if not np.isnan(f)]), X_diff.shape[1], X_diff.shape[2]))
cnt = 0
for i in range(len(regmat[0])):
    if not np.isnan(regmat[0][i]):
        X_in[cnt, :, :] = X_diff[i, :,:]
        cnt += 1

regmat = regmat[:, ~np.isnan(regmat).any(axis=0)]# dropnans

regmat = np.array([scipy.stats.zscore(i) for i in regmat])

# load in data to GLM data struture
dat = glmtools.data.TrialGLMData(data=X_in, dim_labels=['participants', 'time', 'channels'])

# add regressor for intercept
regs = list()
regs.append(glmtools.regressors.ConstantRegressor(num_observations=X_in.shape[0]))
# make contrasts and add intercept
contrasts = [glmtools.design.Contrast(name='Intercept',values=[1] + [0]*regmat.shape[0])]
#loop through continous regreessors and add (also to the info and contrasts)
for i in range(regmat.shape[0]):
    regs.append(glmtools.regressors.ParametricRegressor(values=regmat[i],
                                                    name=names_to_regress[i],
                                                    preproc='z',
                                                    num_observations=X_in.shape[0]))
    # add covariate to info
    dat.info[names_to_regress[i]] = regmat[i]
    values = [0] * (regmat.shape[0] +1)
    values[i+1] = 1
    contrasts.append(glmtools.design.Contrast(name=names_to_regress[i],values=values))

# contrasts
des = glmtools.design.GLMDesign.initialise(regs,contrasts)


model = glmtools.fit.OLSModel( des, dat )

#%% plot all the components
stats = model.get_tstats()
stats = np.abs(stats)
# Plot the outputs
for i in range(stats.shape[0]):
    name = model.contrast_names[i]
    dum_ev = mne.EvokedArray(np.transpose(stats[i,:,:]), info=words_e[0].info, tmin=-0.3,
                        nave=len(model.design_matrix))
    dum_ev.plot_joint(title=name, ts_args=dict(time_unit='s'),
                                 topomap_args=dict(time_unit='s')).savefig('/imaging/ai05/images/glm_' + name+ '_tstats.png')

#%% try and get a cluster thing getting sensible clusters
from mne.stats.cluster_level import _find_clusters, _reshape_clusters

c_thresh = 2
tstats = model.get_tstats()
#tstats = model.copes
flatt = tstats[4].flatten()
flatt = np.abs(flatt)
connectivity, ch_names = mne.channels.find_ch_connectivity(words_e[0].info, ch_type='mag')
connectivity = mne.stats.cluster_level._setup_connectivity(connectivity, len(flatt), tstats.shape[1])


# clus is a 1d mask and stat is the associated tstat
clus, cstat = _find_clusters(flatt, threshold=c_thresh, tail=1, connectivity=connectivity)
clusters = _reshape_clusters(clus, (tstats[0].shape[0], tstats[0].shape[1]))

#%% permute for cluster distribution
for f in des.regressor_list:
    f.rtype = 'Parametric'
#clus_null = red_perm.c_corrected_permute_glm(des, dat, nperms=100, threshold=3)
clus_null = cluster_c_corrected_permute_glm(des, dat, nperms=1000, threshold=c_thresh,
                                                    connectivity=connectivity,
                                                    stat='cope',
                                                     scriptdir='/imaging/ai05/phono_oddball/cluster_scripts',
                                                     pythondir='/home/ai05/.conda/envs/mne_2/bin/python',
                                                     filesdir='/imaging/ai05/phono_oddball/cluster_files')
clus_null = clus_null[:,1:-1]
np.save('/imaging/ai05/phono_oddball/permuted_clusters_1000_cope_abs.npy', clus_null)
#%%
clus_null = np.load('/imaging/ai05/phono_oddball/permuted_clusters_1000_tstat_abs.npy')
#%% calculate pluster p values based on null values

import scipy.stats as ss
all_clus = []
p_vals = []
stats = np.abs(tstats)
ntimes = stats[0].shape[0]
nchans = stats[0].shape[1]
for i in range(stats.shape[0]):
    #reshape the scores
    ins = stats[i].reshape((ntimes*nchans))
    all_clus.append(_find_clusters(ins, threshold=c_thresh, tail=1, connectivity=connectivity))
    this_c, this_s = all_clus[i]
    tmp_score = [ss.percentileofscore(clus_null[i], f)/100 for f in this_s]
    p_vals.append(tmp_score)

sig_clusts_mask = [np.array(f) > 0.95 for f in p_vals]

# reshape clusters
all_cs = [f[1] for f in all_clus] # cluster stats
all_c = [f[0] for f in all_clus] # locations
all_c = [_reshape_clusters(f, (ntimes, nchans)) for f in all_c] # reshape locations back to 2D

sig_clusters = [np.array(all_c[i])[sig_clusts_mask[i]] for i in range(len(all_c))]

#%% plot the cluster
cont = 4
t_vals = stats[cont]
time_inds, space_inds = sig_clusters[cont][0]

ch_inds = np.unique(space_inds)
time_inds = np.unique(time_inds)

# get topography of t stat
t_map = t_vals[time_inds, :].mean(axis=0)
# get signals at the sensors contributing to the cluster
times =  words_e[0].times
sig_times = times[time_inds]
# create spatial mask
mask = np.zeros((t_map.shape[0], 1), dtype=bool)
mask[ch_inds, :] = True

# initialize figure
fig, ax_topo = plt.subplots(1, 1, figsize=(10, 3))


pos = mne.find_layout(words_e[0].info).pos
# plot average test statstic and mark significant sensors
image, _ = mne.viz.plot_topomap(t_map, pos, mask=mask, axes=ax_topo, cmap='Reds',
                        vmin=np.min, vmax=np.max, show=False)

# create additional axes (for ERF and colorbar)
divider = make_axes_locatable(ax_topo)

# add axes for colorbar
ax_colorbar = divider.append_axes('right', size='5%', pad=0.05)
plt.colorbar(image, cax=ax_colorbar)
ax_topo.set_xlabel(
    'Averaged T-map ({:0.3f} - {:0.3f} s)'.format(*sig_times[[0, -1]]))

# add new axis for time courses and plot time courses
ax_signals = divider.append_axes('right', size='300%', pad=1.2)
title = 'Cluster #{0}, {1} sensor'.format(1, len(ch_inds))
if len(ch_inds) > 1:
    title += "s (mean)"

#make dummy evoked
dum_ev = mne.EvokedArray(np.transpose(t_vals), info=words_e[0].info, tmin=-0.3,
                             nave=len(regmat[0]))

mne.viz.plot_compare_evokeds(dum_ev,
                     title=title,
                     picks=ch_inds,
                     combine='mean',
                     axes=ax_signals,
                     show=False,
                     split_legend=True,
                     truncate_yaxis='max_ticks')

# plot temporal cluster extent
ymin, ymax = ax_signals.get_ylim()
ax_signals.fill_betweenx((ymin, ymax), sig_times[0], sig_times[-1],
                         color='orange', alpha=0.3)
ax_signals.set_ylabel('Cluster Mean T')


#dum_ev.plot(axes=big_ax[1])
# clean up viz
fig.subplots_adjust(bottom=.05)
fig.savefig(f'/imaging/ai05/images/cope_cluster_1_{cont}.png')
plt.close('all')

fig, ax_topo = plt.subplots(1, 1, figsize=(10, 3))
# on second subplot plot t-value of all sensors
dum_ev.plot_joint( picks=ch_inds).savefig(f'/imaging/ai05/images/cope_{cont}.png')

plt.close('all')
#%%
dum_ev_reg = mne.EvokedArray(np.transpose(tstats[0]), info=words_e[0].info, tmin=-0.3,
                             nave=len(regmat[0]))

plt.close('all')
plt.imshow(clusters[0])
plt.show()

#%% put the copes / betas into an evoked object

stats = np.abs(tstats)
dum_ev_int = mne.EvokedArray(np.transpose(stats[0,:,:]), info=words_e[0].info, tmin=-0.3,
                        nave=len(regmat[0]))
dum_ev_int.plot_joint().savefig('/imaging/ai05/images/glm_intercept_cope.png')
for i in range(stats.shape[0]):
    dum_ev_reg = mne.EvokedArray(np.transpose(stats[i+1,:,:]), info=words_e[0].info, tmin=-0.3,
                        nave=len(regmat[0]))

    dum_ev_reg.plot_joint().savefig(f'/imaging/ai05/images/glm_{names_to_regress[i]}_cope.png')


#%% get null distributions
# note that the first row will be the original model, unpermuted
for f in des.regressor_list:
    f.rtype = 'Parametric'
    #%%
permuted = red_perm.permute_glm(des, dat, nperms=1, stat='copes',nomax_axis=None)
# %% Try the cluster one
permuted = red_perm.permute_glm_cluster(des, dat, nperms=1000, stat='cope',nomax_axis=None,
                                        scriptdir='/imaging/ai05/phono_oddball/cluster_scripts',
                                        pythondir='/home/ai05/.conda/envs/mne_2/bin/python',
                                        filesdir='/imaging/ai05/phono_oddball/cluster_files')
#%%
np.save('/imaging/ai05/phono_oddball/permuted_factors_copes.npy', permuted)
#%%
permuted = np.load('/imaging/ai05/phono_oddball/permuted_factors_copes.npy')
 #%%
"""
output from perm 
"""

thresh = 0.1
# make empty matrix for holding permute percentiles
# contrast x low/hi x timepoint

# extract actual model from first perm
stats = permuted[:,0,:,:]

permutations = permuted[:,1:-1,:,:]

#%% signed
# get mask of significant time points relative to model
threshold_perms_upper = np.percentile(permutations[:,:,:,:], [100-thresh], axis=1)
threshold_perms_lower = np.percentile(permutations[:,:,:,:], [thresh], axis=1)
# force any cope stat in the image with a False to be 0

mask = np.where((stats < threshold_perms_upper[0,:,:,:]) & (stats > threshold_perms_lower[0,:,:,:]), False, True)# mask for all

masked_model = stats.copy()
masked_model[~mask] = 0

#%% abs
# or absolute sign
permutations = np.abs(permutations)
stats = np.abs(stats)
threshold_abs = np.percentile(permutations[:,:,:,:], [100-thresh], axis=1)
mask = (stats > threshold_abs[0,:,:,:])

masked_model = stats.copy()
masked_model[~mask] = 0
#%% plot the results of permutation test

# Plot the outputs
for i in range(stats.shape[0]):
    name = model.contrast_names[i]
    dum_ev = mne.EvokedArray(np.transpose(stats[i,:,:]), info=words_e[0].info, tmin=-0.3,
                        nave=len(model.design_matrix))
    dum_ev.plot_joint(title=name, ts_args=dict(time_unit='s'),
                                 topomap_args=dict(time_unit='s')).savefig('/home/ai05/reg_old_' + name+ '.png')
    #
    # reject_H0 = np.transpose(mask[i, :,:])
    # dum_ev.plot_image(mask=reject_H0, time_unit='s').savefig('/home/ai05/reg_p_vals_old_' +  name+ '.png')



