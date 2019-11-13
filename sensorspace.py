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

#%% Get all evoked files

#%% source estimates for all evoked responses -- setup files
# get all evoked files in folder
all_evo = [f for f in os.listdir(mne_evo_out) if os.path.isfile(f'{mne_evo_out}/{f}')]
# make sure they are concat files
all_evo = [f for f in all_evo if 'concat' in f]

# get numbers of participants
nums = set([f.split('_')[0] for f in all_evo])
words = [f'{f}_concat_MNN-Word_ave.fif' for f in nums] # file names for words
words = [f for f in words if os.path.isfile(mne_evo_out+'/'+f)]
non_words = [f'{f}_concat_MNN-Non-Word_ave.fif' for f in nums] # file names for non_words
non_words = [f for f in non_words if os.path.isfile(mne_evo_out+'/'+f)]
#check matches
nonmatches = [w.split('_')[0] != n.split('_')[0] for (w, n) in zip(words, non_words)]
print('are there any non-matching entries:')
print(any(nonmatches))

#%%
words_e = []
for ef in words:
    ev = mne.read_evokeds(ef)
    words_e.append(ev[0])

non_words_e = []
for ef in non_words:
    ev = mne.read_evokeds(ef)
    non_words_e.append(ev[0])
#%%
fig = mne.viz.plot_compare_evokeds({'words':words_e[20], 'non-words':non_words_e[20]}, split_legend=True,
                             axes="topo")
fig[0].savefig(f'/home/ai05/comp_top_ev_mag.png')
fig[1].savefig(f'/home/ai05/comp_top_ev_grad.png')
#%%
fig = mne.viz.plot_compare_evokeds({'words':words_e, 'non-words':non_words_e},
                                   picks=['MEG2221', 'MEG2441', 'MEG2411', 'MEG2431'],
                                   combine='mean')
fig[0].savefig(f'/home/ai05/comp_ev.png')
#fig[1].savefig(f'/home/ai05/comp_ev_1.png')

#%%
comb = mne.combine_evoked(words_e, [1]*len(words_e))
fig = comb.plot(picks=['MEG2221', 'MEG2441', 'MEG2411', 'MEG2431'])
fig.savefig(f'/home/ai05/grnd_0.png')
#fig[0].savefig(f'/home/ai05/grnd_0.png')
#fig[1].savefig(f'/home/ai05/grnd_1.png')