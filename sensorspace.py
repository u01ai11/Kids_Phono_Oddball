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
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd
import scipy
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

# get behavioural factors
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
#%% Look at topology for mag and grad
fig = mne.viz.plot_compare_evokeds({'words':words_e[20], 'non-words':non_words_e[20]}, split_legend=True,
                             axes="topo")
fig[0].savefig(f'/home/ai05/comp_top_ev_mag.png')
#fig[1].savefig(f'/home/ai05/comp_top_ev_grad.png')
#%% based on this choose some sensors and plot comparisons
fig = mne.viz.plot_compare_evokeds({'words':words_e, 'non-words':non_words_e},
                                   picks=['MEG2221', 'MEG2441', 'MEG2411', 'MEG2431'],
                                   combine='mean')
fig[0].savefig(f'/home/ai05/comp_ev.png')


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

#%% run
connectivity = mne.channels.find_ch_connectivity(words_e[0].info, ch_type='mag')
threshold = 5.0  # very high, but the test is quite sensitive on this data
# set family-wise p-value
p_accept = 0.01

cluster_stats = mne.stats.spatio_temporal_cluster_test(X, n_permutations=1000,
                                             threshold=threshold, tail=1,
                                             n_jobs=1, buffer_size=None,
                                             connectivity=connectivity[0])
#%% Do we have any good clusters?
T_obs, clusters, p_values, _ = cluster_stats
good_cluster_inds = np.where(p_values < p_accept)[0]

#%% plot them
colors = {"Dev Word": "crimson", "Dev Non-Word": 'steelblue'}
linestyles = {"L": '-', "R": '--'}

# get sensor positions via layout
pos = mne.find_layout(words_e[0].info).pos

# combine evokeds for group level
word = mne.combine_evoked(words_e, weights=[1]*len(words_e))
non_word = mne.combine_evoked(non_words_e, weights=[1]*len(non_words_e))
evokeds = {'Dev Word': word, 'Dev Non-Word': non_word}

# list to keep masks in for later analysis
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
        'Averaged F-map ({:0.3f} - {:0.3f} s)'.format(*sig_times[[0, -1]]))

    # add new axis for time courses and plot time courses
    ax_signals = divider.append_axes('right', size='300%', pad=1.2)
    title = 'Cluster #{0}, {1} sensor'.format(i_clu + 1, len(ch_inds))
    if len(ch_inds) > 1:
        title += "s (mean)"
    mne.viz.plot_compare_evokeds(evokeds, title=title, picks=ch_inds, axes=ax_signals,
                         colors=colors, show=False,
                         split_legend=True, truncate_yaxis='auto', combine='mean')

    # plot temporal cluster extent
    ymin, ymax = ax_signals.get_ylim()
    ax_signals.fill_betweenx((ymin, ymax), sig_times[0], sig_times[-1],
                             color='orange', alpha=0.3)

    # clean up viz
    mne.viz.tight_layout(fig=fig)
    fig.subplots_adjust(bottom=.05)

    fig.savefig(f'/home/ai05/cluster_{str(i_clu)}.png')


#%% do thesame but with a custom function for the statistics
# make dummy epochs object with group level data
# each epoch = 1 participant's difference
X_diff = X[0] - X[1]

# as we are doing a behaviour covariate regression exclude data we have no covariate for
facts = pd.read_csv('/imaging/ai05/phono_oddball/ref_data/components_MNE.csv')
facts = facts.to_numpy()

WMfact = [f[5] for f in facts]
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
names = ['Intercept', 'Education fact']
# Now run the linear regression
res = mne.stats.linear_regression(epoch, design_mat, names)

# Plot the outputs
for name in names:
    res[name].beta.plot_joint(title=name, ts_args=dict(time_unit='s'),
                                 topomap_args=dict(time_unit='s')).savefig('/home/ai05/reg_' + name+ '.png')

    reject_H0, fdr_pvals = mne.stats.fdr_correction(res[name].p_val.data)
    evoked = res[name].beta
    evoked.plot_image(mask=reject_H0, time_unit='s').savefig('/home/ai05/reg_p_vals' + name + '.png')
#%%

#%%
# the spatio temporal cluster_permutation function feeds in a flattened array
# we need to get this data into a format where we can do a regression
# for each cluster the test gets input:
# participant x conditions x timepoints
def stat_fun(*args):
    """
    :param args:
    :return:
    """

    #X_diff = X[0] - X[1]

    # as we are doing a behaviour covariate regression exclude data we have no covariate for
    facts = pd.read_csv('/imaging/ai05/phono_oddball/ref_data/components_MNE.csv')
    facts = facts.to_numpy()

    WMfact = [f[0] for f in facts]
    X_in = np.empty((len([f for f in WMfact if not np.isnan(f)]), X_diff.shape[1], X_diff.shape[2]))
    cnt = 0
    for i in range(len(WMfact)):
        if not np.isnan(WMfact[i]):
            X_in[cnt, :, :] = X_diff[i, :, :]
            cnt += 1

    WMfact_clean = [f for f in WMfact if not np.isnan(f)]

    epoch = mne.read_epochs(mne_epo_out + '/' + os.listdir(mne_epo_out)[0])
    epoch.pick('mag')
    epoch.event_id = {'MNN Difference': 1}
    epoch._data = np.transpose(X_in, (0, 2, 1))

    time = 300
    epoch.events = epoch.events[0:len(X_in)]
    for i in range(len(X_in)):
        epoch.events[i] = [time, 0, 1]
        time += 1100

    # design matrix
    design_mat = np.transpose(np.array(([1] * len(epoch), WMfact_clean)))