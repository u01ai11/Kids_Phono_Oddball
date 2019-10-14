import mne
import os
import numpy as np
import joblib


def setup_src_multiple(sublist, fs_sub_dir, outdir, spacing, surface, n_jobs1, n_jobs2):

    """
    :param sub:
        Subjects in the freesurfer subjectdir to use
    :param fs_sub_dir:
        Freesurfer subject dir
    :param outdir:
        Where the models are being saved
    :param spacing:
        Input for MNE setup_source_space command. The spacing to use.
    :param surface:
        Input for MNE setup_source_space command. The surface to use
    :param n_jobs1:
        First parallel command, sets the number of joblib jobs to do on group level.
    :param njobs2:
        Second parallel command, sets the number of joblib jobs to submit on the participant level
    :return:
    """

    # set up

    # check if dir exists, make if not

    saved_files = []

    if n_jobs1 == 1:
        for i in range(len(sublist)):
            savedfile = __setup_src_individual(sublist[i], fs_sub_dir, outdir, spacing, surface, n_jobs2)
            saved_files.append(savedfile)
    if n_jobs1 > 1:

        saved_files = joblib.Parallel(n_jobs =n_jobs1)(
            joblib.delayed(__setup_src_individual)(thisS, fs_sub_dir, outdir, spacing, surface, n_jobs2) for thisS in sublist)

    return saved_files


def __setup_src_individual(sub, fs_sub_dir, outdir, spacing, surface, njobs):
    """
    :param sub:
        subject to set up source-space on
    :param fs_sub_dir:
        where to find the fs recon-all files
    :param spacing:
        spacing to use for source-space
    :param surace:
        surface to use for source-space
    :param njobs:
        how many jons to split this up into
    :return:
    """
    src_space = mne.setup_source_space(sub, spacing=spacing, surface=surface, subjects_dir=fs_sub_dir, n_jobs=njobs)
    fname = outdir + '/' + sub + '_'+surface+'-'+spacing+'-src.fif'
    mne.write_source_spaces(fname, src_space)  # write to source dir
    this_sub_dir = fname
    return this_sub_dir


