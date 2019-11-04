import mne
import os
from os.path import dirname
import numpy as np
import joblib
import inspect

def src_group_average(hemi_list, hemis):
    """
    :param hemi_list:
        a list of pairs for the stc files relating to each hemisphere
    :return:
        a path to the averaged file
    """

    avlist = []

    for pair in hemi_list:
        if hemis == 'rh':
            est = mne.read_source_estimate(pair[0], 'fsaverage')
        if hemis == 'lh':
            est = mne.read_source_estimate(pair[1], 'fsaverage')
        if hemis == 'both':
            est = mne.read_source_estimate(pair[0][0:-7], 'fsaverage')
        avlist.append(est)

    average = avlist[0].copy()
    nofiles = len(avlist)
    for i in range(1, nofiles):
        average._data += avlist[i].data

    average._data /= nofiles

    return average



