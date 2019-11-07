import mne
import os
from os.path import dirname
import numpy as np
import joblib
import inspect

def src_concat_mat(stc_lists, morph2):
    """
    :param stc_lists:
        a list of file lists. They must all be the same length
    :param morph2:
        the fs subject we want to morph other files onto
    :return X:
        Returns a matrix with shape:
        space x time x subjects x condition
    """

    #loop through each condition
    est_list =[]
    for i in range(len(stc_lists)):
        est_list.append([]) # empty storage
        # loop through files
        for pair in stc_lists[i]:
            #read and morph
            est = mne.read_source_estimate(pair[0:-7], morph2)
            # append
            est_list[i].append(est)

    # make empty matrix to store
    # space x time x subjects x condition
    X = np.empty((est_list[0][0].shape[0], est_list[0][0].shape[1], len(est_list[0]), len(est_list)))

    # loop through parts and add in data
    for con in range(len(stc_lists)):
        for part in range(len(stc_lists[con])):
            this_est = est_list[con][part]
            X[:, :, part, con] = this_est.data

    return X

def submit_cluster_perm(Xf, srcf, jobs, buffer, t_thresh, scriptpath, pythonpath, outpath):
    """
    :param Xf:
        path to .npy difference matrix with dimensions  samples (subjects) x time x space,
    :param srcf:
        file for source space to make connectivity mat from
    :param jobs:
        how many jobs to split up into
    :param buffer:
        memory buffer for jobs
    :param t_thresh:
        critical t value for clusters
    :param p_thresh:
        critical p value for clusters
    :return:
        filepath to saved output
    """

    pycom = f"""
import mne 
import numpy as np

X = np.load('{Xf}')
src = mne.read_source_spaces('{srcf}')
connectivity = mne.spatial_src_connectivity(src)

clu = \
    mne.stats.spatio_temporal_cluster_1samp_test(X, connectivity=connectivity, n_jobs={jobs},
                                       threshold={t_thresh}, buffer_size={buffer},
                                       verbose=True)
np.save('{outpath}/clus_results', clu)
    """

    # save to file
    print(pycom, file=open(f'{scriptpath}/batch_clustperm.py', 'w'))

    # construct csh file
    tcshf = f"""#!/bin/tcsh
    {pythonpath} {scriptpath}/batch_clustperm.py.py
            """
    # save to directory
    print(tcshf, file=open(f'{scriptpath}/batch_clustperm.csh', 'w'))

    # execute this on the cluster
    os.system(f'sbatch --job-name=clusterfu --mincpus=28 -t 0-3:00 {scriptpath}/batch_clustperm.csh')
