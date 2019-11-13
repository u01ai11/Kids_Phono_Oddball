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

#%% get files
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

fwds = [f'{f}-volume-fwd.fif' for f in nums]
fwds = [f for f in fwds if os.path.isfile(mne_src_dir+'/'+f)]

epochs = [f'{f}_concat_epo.fif' for f in nums]
epochs = [f for f in epochs if os.path.isfile(mne_epo_out+'/'+f)]

#%% TRY LCMV on ONE
i = 0
epoch =  mne.read_epochs(f'{mne_epo_out}/{epochs[i]}')
word_ev = mne.read_evokeds(f'{mne_evo_out}/{words[i]}')
fwd = mne.read_forward_solution(f'{mne_src_dir}/{fwds[i]}')

noise_cov = mne.compute_covariance(epoch, tmin=-3, tmax=0, method='empirical',
                                   rank=None)
data_cov = mne.compute_covariance(epoch, tmin=0.00, tmax=0.8,
                                  method='empirical', rank=None)

filters = mne.beamformer.make_lcmv(word_ev[0].info, fwd, data_cov, reg=0.05,
                    noise_cov=noise_cov, pick_ori='max-power',
                    weight_norm='nai', rank=None)
stc = mne.beamformer.apply_lcmv(word_ev[0], filters, max_ori_out='signed')

#%%
lims = [0.3, 0.6, 0.9]
stc.plot(
    src=fwd['src'], subjects_dir=fs_sub_dir,
    clim=dict(kind='value', pos_lims=lims), mode='stat_map',
    initial_time=0.7, verbose=True).savefig('/home/ai05/voltest.png')
#%%

#do it all

red_inv.lcmv_multiple(epoch_list=[f'{mne_epo_out}/{f}' for f in epochs],
                      evoked_list=[f'{mne_evo_out}/{f}' for f in words],
                      forward_list=[f'{mne_src_dir}/{f}' for f in fwds],
                      fs_sub_dir=fs_sub_dir,
                      scriptpath='/home/ai05/clusterscripts',
                      pythonpath='/home/ai05/anaconda3/envs/mne_2/bin/python',
                      outpath='/imaging/ai05/phono_oddball/mne_ev_src')
#%%
#check matches
nonmatches = [w.split('_')[0] != n.split('_')[0] for (w, n) in zip(words, non_words)]
print('are there any non-matching entries:')
print(any(nonmatches))


