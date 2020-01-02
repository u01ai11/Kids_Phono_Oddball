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

#%% Get all evoked files

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
#%%
tshift = -0.37
words_e = []
for ef in words:
    ev = mne.read_evokeds(ef)
    ev[0].pick('mag')
    ev[0].data = np.negative(ev[0].data)
    ev[0].shift_time(tshift)
    words_e.append(ev[0])

non_words_e = []
for ef in non_words:
    ev = mne.read_evokeds(ef)
    ev[0].pick('mag')
    ev[0].data = np.negative(ev[0].data)
    ev[0].shift_time(tshift)
    non_words_e.append(ev[0])
#%% Look at topology for mag and grad
fig = mne.viz.plot_compare_evokeds({'words':words_e, 'non-words':non_words_e}, split_legend=True,
                             axes="topo")
fig[0].savefig(f'/home/ai05/comp_top_ev_mag.png')
#fig[1].savefig(f'/home/ai05/comp_top_ev_grad.png')
#%% based on this choose some sensors and plot comparisons
colors = {"Words": "crimson", "Non-Words": 'steelblue'}

right_parietal_temporal = ['MEG1131', 'MEG1341', 'MEG1331', 'MEG2221', 'MEG2411','MEG2421','MEG2441','MEG2231',
                           'MEG2431']
right_temporal = ['MEG1311','MEG1321','MEG1441','MEG1421','MEG1431','MEG1341','MEG1331','MEG2611','MEG2621',
                  'MEG2641','MEG2421','MEG2411','MEG2631']

fig, ax = plt.subplots(1, 1, figsize=(9, 4))

fig2 = mne.viz.plot_compare_evokeds({'Words':words_e, 'Non-Words':non_words_e},
                                    picks=right_parietal_temporal,
                                    combine='mean',
                                    axes=ax,
                                    show_sensors=True,
                                    colors=colors,

                                    title='Evoked response to Words vs Non-Words')

# plot lines with points on them
ylims = ax.get_ylim()
ax.axvline(x=tshift)
ax.axvline(x=0)

ax.text(tshift + 0.02, ylims[1] * 0.8, 'Stimuli Start')
ax.text(0 + 0.02, ylims[1] * 0.8, 'Final Phoneme')

ax.set_ylabel('Field Strength (fT)')

fig.savefig(f'/imaging/ai05/images/comp_ev.png', dpi=500)

#%% topology
# combine evokeds for group level
word = mne.combine_evoked(words_e, weights=[1]*len(words_e))
non_word = mne.combine_evoked(non_words_e, weights=[1]*len(non_words_e))
lists = {'Dev Word': word, 'Dev Non-Word': non_word}
evokeds = lists.copy()
# list to keep masks in for later analysis
reg_masks = []

titles = ['Words', 'Non-Words']
times = np.arange(-0.1, 0.6, 0.1)
fig, axes = plt.subplots(figsize=(7.5, 2.5), nrows=2)
for i in range(len(evokeds)):
    key = list(evokeds.keys())[i]
    evokeds[key].plot_topomap(times, ch_type='mag', time_unit='s',
                              title=titles[i]).savefig(f'/imaging/ai05/images/topomaps_evoked{key}.png')


#%% Get data into acceptable format for cluster permutation testing
# This is just a one-way F test to compare the two word/nonword MNN
# If signiticant we will move onto behavioural covariants

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

# Do not flip as we did this above already

#%% We first need to sort out connectivity
# This will be a spatio-temporal connectivity matrix
connectivity2 = mne.channels.find_ch_connectivity(words_e[0].info, ch_type='mag')

#%%
threshold = 4  # very high, but the test is quite sensitive on this data
# set family-wise p-value
p_accept = 0.05

cluster_stats = mne.stats.spatio_temporal_cluster_1samp_test((X[0]-X[1]), n_permutations=5000,
                                             threshold=threshold, tail=0,
                                             n_jobs=10, buffer_size=None,
                                             connectivity=connectivity2[0])
#%% Do we have any good clusters?
T_obs, clusters, p_values, _ = cluster_stats
good_cluster_inds = np.where(p_values < p_accept)[0]

#%% plot them
colors = {"Words": "crimson", "Non-Words": 'steelblue'}
linestyles = {"L": '-', "R": '--'}

# get sensor positions via layout
pos = mne.find_layout(words_e[0].info).pos


#%%
reg_masks = []
# loop over clusters
for i_clu, clu_idx in enumerate(good_cluster_inds):
    # unpack cluster information, get unique indices
    time_inds, space_inds = np.squeeze(clusters[clu_idx])
    ch_inds = np.unique(space_inds)
    time_inds = np.unique(time_inds)

    # get topography for F stat
    f_map = T_obs[time_inds, ...].mean(axis=0)

    # get signals at the sensors contributing to the cluster
    sig_times = words_e[0].times[time_inds]

    # create spatial mask
    mask = np.zeros((f_map.shape[0], 1), dtype=bool)
    mask[ch_inds, :] = True
    reg_masks.append(mask)
    # initialize figure
    fig, ax_topo = plt.subplots(1, 1, figsize=(10, 3))

    # plot average test statistic and mark significant sensors
    image, _ = mne.viz.plot_topomap(f_map, pos, mask=mask, axes=ax_topo, cmap='Reds',
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
    title = 'Cluster #{0}, {1} sensor'.format(i_clu + 1, len(ch_inds))
    if len(ch_inds) > 1:
        title += "s (mean)"
    mne.viz.plot_compare_evokeds({'Words':words_e, 'Non-Words':non_words_e}, title=title, picks=ch_inds, axes=ax_signals,
                         colors=colors, show=False,
                         split_legend=True, truncate_yaxis='auto', combine='mean')

    # plot temporal cluster extent
    ymin, ymax = ax_signals.get_ylim()
    ax_signals.fill_betweenx((ymin, ymax), sig_times[0], sig_times[-1],
                             color='orange', alpha=0.3)

    ylims = ax_signals.get_ylim()
    ax_signals.axvline(x=tshift)
    ax_signals.axvline(x=0)
    ax_signals.text(tshift + 0.02, ylims[1] * 0.8, 'Stimuli Start')
    ax_signals.text(0 + 0.02, ylims[1] * 0.8, 'Final Phoneme')

    ax_signals.set_ylabel('Field Strength (fT)')
    # clean up viz
    mne.viz.tight_layout(fig=fig)
    fig.subplots_adjust(bottom=.05)

    fig.savefig(f'/imaging/ai05/images/new_cluster_{str(i_clu)}.png')


#%% do thesame but with a custom function for the statistics
# make dummy epochs object with group level data
# each epoch = 1 participant's difference
X_diff = X[0]
# as we are doing a behaviour covariate regression exclude data we have no covariate for
facts = pd.read_csv('/imaging/ai05/phono_oddball/ref_data/components_MNE.csv')
facts = facts.to_numpy()

WMfact = [f[2] for f in facts]

X_in = np.empty((len([f for f in WMfact if not np.isnan(f)]), X_diff.shape[1], X_diff.shape[2]))
cnt = 0
for i in range(len(WMfact)):
    if not np.isnan(WMfact[i]):
        X_in[cnt, :, :] = X_diff[i, :,:]
        cnt += 1

WMfact_clean = [f for f in WMfact if not np.isnan(f)]

epoch = mne.read_epochs(mne_epo_out+'/'+os.listdir(mne_epo_out)[0])
epoch.pick('mag')
epoch.event_id = {'MNN Difference': 1}
epoch._data = np.transpose(X_in, (0, 2, 1))

time = 300
epoch.events = epoch.events[0:len(X_in)]
for i in range(len(X_in)):
    epoch.events[i] = [time, 0, 1]
    time += 1100

# optional zscore mat
WMfact_clean = scipy.stats.zscore(WMfact_clean)
#
# design matrix
design_mat = np.transpose(np.array(([1]*len(epoch), WMfact_clean)))

#%% optionally mask off any non-cluster based sensors
drop_mask = reg_masks[0]
# use sensor mask to get sensor names
chan_names = epoch.info['ch_names']
pickchans = []
for i in range(len(chan_names)):
    if drop_mask[i] == True:
        pickchans.append(chan_names[i])
epoch.pick(pickchans)


#%%
names = ['Intercept', 'Factor IQ']
# Now run the linear regression
res = mne.stats.linear_regression(epoch, design_mat, names)

# Plot the outputs
for name in names:
    res[name].beta.plot_joint(title=name, ts_args=dict(time_unit='s'),
                                 topomap_args=dict(time_unit='s')).savefig('/home/ai05/reg_' + name+ '.png')

    reject_H0, fdr_pvals = mne.stats.fdr_correction(res[name].p_val.data)
    evoked = res[name].beta
    evoked.plot_image(mask=reject_H0, time_unit='s').savefig('/home/ai05/reg_p_vals' + name + '.png')
#%% Try this also by summing across sensors for each

drop_mask = [f[0] for f in reg_masks[0]]
# use sensor mask to get sensor names
chan_names = epoch.info['ch_names']
# get array with just cluster channels
X_cluster = X_in[:,:,drop_mask]
#X_cluster =
#gfp
X_gfp = np.std(np.mean(X_cluster, 2),0)
#mean
X_mean = np.mean(X_cluster, (0,2))
#std
X_std = np.std(X_cluster, (0,2))

fig, ax = plt.subplots(1, 2, figsize=(10, 3))
ax[0].plot(np.transpose(X_mean))
ax[0].plot(np.transpose(X_std))
ax[0].set_title( 'Mean')

ax[1].plot(X_gfp)
ax[1].set_title( 'gfp')
fig.savefig('/home/ai05/av.png')

X_group_mean = np.mean(X_cluster, 2)

#%% Use internal function '_fit_lm' to run OLS straight on our matrix
#names = ['Intercept', 'Education fact']

lm_params = mne.stats.regression._fit_lm(X_group_mean, design_mat, names)
lm = namedtuple('lm', 'beta stderr t_val p_val mlog10_p_val')
labels = ['beta', 'stderr', 't_val', 'p_val', 'mlog_10_p_val']

#plot the betas
fig, ax = plt.subplots(len(names),1, figsize=(10, 3))

for i, name in enumerate(names):
    ax[i].plot(lm_params[0][name])
    ax[i].set_title(name + ' beta weights')

fig.savefig('/home/ai05/lm_clusteraverage_betas.png')
fig, ax = plt.subplots(len(names),1, figsize=(10, 3))
for i, name in enumerate(names):
    p_vals = lm_params[3][name]
    reject_H0, corr_p = mne.stats.fdr_correction(p_vals)
    ax[i].plot(corr_p)
    ax[i].set_title(name + ' p_vals')

fig.savefig('/home/ai05/lm_clusteraverage_ps.png')

plt.close('all')

# # output is length of 5: beta, stderr, t_val, p_val and mlog10_p_val
# for i, item in enumerate(['beta', 'stderr', 't_val', 'p_val', 'mlog_10_p_val']):
#


#%% Ite let's try this using Quinn's glmtools library
import glmtools
"""
- Create a data matrix containing each participants MMN [ participants x time x voxel ]
- Add behavioural score to the data.info dict
- Create a design with two regressors
    first is a constant (mean-term)
    second is a parametric regressor containing the z-transformed behavioural score.
    
The fitted GLM will return a [2 x time x voxel] matrix in which [0,:,:] contains the mean MMN and [1,:,:,] contains the covariance between behavioural score and the MMN across participants.

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

#%% try and get a cluster thing getting sensible clusters
from mne.stats.cluster_level import _find_clusters, _reshape_clusters

tstats = model.get_tstats()
flatt = tstats[1].flatten()
connectivity, ch_names = mne.channels.find_ch_connectivity(words_e[0].info, ch_type='mag')
connectivity = mne.stats.cluster_level._setup_connectivity(connectivity, len(flatt), tstats.shape[1])


# clus is a 1d mask and stat is the associated tstat
clus, cstat = _find_clusters(tstats[1].flatten(), threshold=3, tail=1, connectivity=connectivity)
clusters = _reshape_clusters(clus, (tstats[0].shape[0], tstats[0].shape[1]))

#%% permute for cluster distribution
for f in des.regressor_list:
    f.rtype = 'Parametric'
#clus_null = red_perm.c_corrected_permute_glm(des, dat, nperms=100, threshold=3)
clus_null = cluster_c_corrected_permute_glm(des, dat, nperms=1000, threshold=2.5,
                                                    connectivity=connectivity,
                                                     scriptdir='/imaging/ai05/phono_oddball/cluster_scripts',
                                                     pythondir='/home/ai05/.conda/envs/mne_2/bin/python',
                                                     filesdir='/imaging/ai05/phono_oddball/cluster_files')

np.save('/imaging/ai05/phono_oddball/permuted_clusters_1000.npy', clus_null)
#%%
clus_null = np.load('/imaging/ai05/phono_oddball/permuted_clusters_1000.npy')
#%% calculate pluster p values based on null values
import scipy.stats as ss
all_clus = []
p_vals = []
for i in range(tstats.shape[0]):
    all_clus.append(_find_clusters(tstats[i].flatten(), threshold=2.5, tail=1, connectivity=connectivity))
    this_c, this_s = all_clus[i]
    tmp_score = [ss.percentileofscore(clus_null[i], f)/100 for f in this_s]
    p_vals.append(tmp_score)

sig_clusts_mask = [np.array(f) > 0.95 for f in p_vals]


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
permuted = red_perm.permute_glm(des, dat, nperms=100, stat='cope',nomax_axis=None)
# %% Try the cluster one
permuted = red_perm.permute_glm_cluster(des, dat, nperms=1000, stat='cope',nomax_axis=None,
                                        scriptdir='/imaging/ai05/phono_oddball/cluster_scripts',
                                        pythondir='/home/ai05/.conda/envs/mne_2/bin/python',
                                        filesdir='/imaging/ai05/phono_oddball/cluster_files')
np.save('/imaging/ai05/phono_oddball/permuted_factors_copes.npy', permuted)
#%%
permuted = np.load('/imaging/ai05/phono_oddball/permuted_t.npy')
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

    reject_H0 = np.transpose(mask[i, :,:])
    dum_ev.plot_image(mask=reject_H0, time_unit='s').savefig('/home/ai05/reg_p_vals_old_' +  name+ '.png')



#%% Test for shifting triggers
i = 2
filtered_in = [f for f in os.listdir(rawdir) if 'clean_raw' in f]
raw = mne.io.read_raw_fif(os.path.join(rawdir,filtered_in[i]), preload=True)
events = mne.find_events(raw, shortest_event=1)
events[:,0] = events[:,0]+370
picks = mne.pick_types(raw.info, meg=True, eog=True, ecg=True, include=trigchan, exclude='bads')  # select channels
keys = {'Freq': 10, 'Dev Word': 11, 'Dev Non-Word': 12}  # pass in keys
trigchan = 'STI101_up'  # pass in the trigger channel
backup_trigchan = 'STI102'
epochs = mne.Epochs(raw, events, keys, -0.3, 1.0, picks=picks, baseline=(None, 0), preload=True)

keys_keys = [i for i in keys.keys()]
evokeds = [epochs[name].average() for name in keys_keys]
MNN_word = mne.combine_evoked([evokeds[1], -evokeds[0]], weights='equal')
MNN_word.pick_types('mag').plot_joint().savefig('/home/ai05/shifted_test.png')
MNN_word.pick_types('mag').plot_topo().savefig('/home/ai05/shifted_test.png')