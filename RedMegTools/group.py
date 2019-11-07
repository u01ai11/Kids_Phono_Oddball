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



