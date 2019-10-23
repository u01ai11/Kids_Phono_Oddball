import mne
import os
import numpy as np
import joblib


def fwd_solution_multiple(raw_list, trans_list, src_list, bems_list, rawdir, mnedir,outdir, n_jobs):
    """
    Makes forward solution for multiple input subjects
    :param raw_list:
        list of raw files to get infor from
    :param trans_list:
        list of transform files from coreg
    :param src_list:
        list of source-space files
    :param bems_list:
        list of bem solution files
    :param rawdir
        dir containing meg files
    :param mnedir
        dir continaing the other files
    :param outdir:
        where we want to save this data
    :param n_jobs:
        number of parallel jobs to run (1 just does sequentially)
    :return:
    """

    # set up
    saved_files = []

    if n_jobs == 1:
        for i in range(len(raw_list)):
            savedfile = __fwd_individual(rawdir+'/'+raw_list[i], mnedir+'/'+trans_list[i], mnedir+'/'+src_list[i], mnedir+'/'+bems_list[i], outdir)
            saved_files.append(savedfile)
    if n_jobs > 1:
        saved_files = joblib.Parallel(n_jobs =n_jobs)(
            joblib.delayed(__fwd_individual)(rawdir+'/'+R, mnedir+'/'+T, mnedir+'/'+S, mnedir+'/'+B, outdir) for R,T,S,B in zip(raw_list, trans_list, src_list, bems_list))

    return saved_files

def __fwd_individual(raw, trans, src, bemsol, outdir):
    """
    private function for creating the forward solution

    :param raw:
        meg file to get data from
    :param trans:
        transform file
    :param src:
        sourcespace file
    :param bemsol:
        bem solution file
    :param outdir:
        the directory to save these files to
    :return:
    """
    f_only = os.path.basename(raw).split('_')  # get filename parts seperated by _
    num = f_only[0]

    # check if file exists
    if os.path.isfile(f'{outdir}/{num}-fwd.fif'):
        print(f'{outdir}/{num}-fwd.fif exists already skipping' )
        return f'{outdir}/{num}-fwd.fif'

    rawf = mne.io.read_raw_fif(raw, preload=False)
    srcf = mne.read_source_spaces(src)
    bemsolf = mne.read_bem_solution(bemsol)
    fwd = mne.make_forward_solution(rawf.info, trans, srcf, bemsolf)
    mne.write_forward_solution(f'{outdir}/{num}-fwd.fif', fwd)

    return f'{outdir}/{num}-fwd.fif'


def cov_matrix_multiple(epochlist, method, rank, outdir, njobs):

    # set up
    saved_files = []

    if n_jobs == 1:
        for i in range(len(epochlist)):
            savedfile = __fwd_individual(rawdir+'/'+raw_list[i], mnedir+'/'+trans_list[i], mnedir+'/'+src_list[i], mnedir+'/'+bems_list[i], outdir)
            saved_files.append(savedfile)
    if n_jobs > 1:
        saved_files = joblib.Parallel(n_jobs =n_jobs)(
            joblib.delayed(__fwd_individual)(rawdir+'/'+R, mnedir+'/'+T, mnedir+'/'+S, mnedir+'/'+B, outdir) for R,T,S,B in zip(raw_list, trans_list, src_list, bems_list))

    return saved_files

def __