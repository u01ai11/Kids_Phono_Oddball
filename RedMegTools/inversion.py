import mne
import os
from os.path import dirname
import numpy as np
import joblib
import inspect

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


def cov_matrix_multiple_cluster(epochlist, method, rank, tmax, outdir, pythonpath, scriptpath):
    """
    Same as below but uses the cluster to submit individual jobs

    :param epochlist:
        list of epoched file to calculate covariance matrix from
    :param method:
        method to use on calculating, see MNE documentation for options
    :param rank:
        rank for MNE covariance function
    :param tmax:
        the time which we want to use from each epoch to go up to
    :param outdir:
        where are we saving
    :param pythonpath
        path to your python excecutable
    :param scriptpath
        path where you want to save cluster scripts
    :param njobs:
        how parallel we need to go
    :return:
    """

    # get covariance matrix function as a string
    cvfunstr = inspect.getsource(__cov_matrix_individual)
    # get path of this library
    parent_path = dirname(dirname(os.path.abspath(__file__)))
    # loop through files
    for i in range(len(epochlist)):
        # construct python file for this subject
        command = f"""
import mne
import os
import numpy as np
import joblib
import inspect
import sys
sys.path.insert(0, '{parent_path}')
{cvfunstr}
__cov_matrix_individual('{epochlist[i]}', '{method}', {rank}, {tmax}, '{outdir}')
        """
        # save to file
        print(command, file=open(f'{scriptpath}/batch_{i}.py', 'w'))

        # construct csh file
        tcshf = f"""#!/bin/tcsh
{pythonpath} {scriptpath}/batch_{i}.py
        """
        # save to directory
        print(tcshf, file=open(f'{scriptpath}/batch_{i}.csh', 'w'))

        # execute this on the cluster
        os.system(f'sbatch --job-name=covmat_{i} --mincpus=5 -t 0-1:00 {scriptpath}/batch_{i}.csh')



def cov_matrix_multiple(epochlist, method, rank, tmax, outdir, njobs):
    """

    :param epochlist:
        list of epoched file to calculate covariance matrix from
    :param method:
        method to use on calculating, see MNE documentation for options
    :param rank:
        rank for MNE covariance function
    :param tmax:
        the time which we want to use from each epoch to go up to
    :param outdir:
        where are we saving
    :param njobs:
        how parallel we need to go
    :return:
    """
    # set up
    saved_files = []

    if njobs == 1:
        for i in range(len(epochlist)):
            savedfile = __cov_matrix_individual(epochlist[i], method, rank, tmax, outdir)
            saved_files.append(savedfile)
    if njobs > 1:
        saved_files = joblib.Parallel(n_jobs =njobs)(
            joblib.delayed(__cov_matrix_individual)(thisepoch, method, rank, tmax, outdir) for thisepoch in epochlist)

    return saved_files


def __cov_matrix_individual(epochf, method, rank, tmax, outdir):
    """
    private function that uses the inputs listed in multiple function

    :param epochf:
    :param method:
    :param rank:
    :param tmax:
    :param outdir:
    :return:
    """
    nameonly = os.path.basename(epochf)
    parts = nameonly.split('_')
    outname = f'{outdir}/{parts[0]}_{parts[1]}-cov.fif'

    if os.path.isfile(outname):
        print(f'{outname} already exists skipping')
        return outname

    epochs = mne.read_epochs(epochf)
    cov = mne.compute_covariance(epochs[0], rank=rank, method=method, tmax=tmax)
    mne.write_cov(outname, cov)
    return outname


def inv_op_multiple(infofs, fwdfs, covfs, loose, depth, outdir, njobs):
    """
    :param infofs:
    :param fwdfs:
    :param covfs:
    :param loose:
    :param depth:
    :param outdir:
    :param njobs:
    :return:
    """

    saved_files = []

    if njobs == 1:
        for i in range(len(infofs)):
            savedfile = __inv_op_individual(infofs[i], fwdfs[i], covfs[i], loose, depth, outdir)
            saved_files.append(savedfile)
    if njobs > 1:
        saved_files = joblib.Parallel(n_jobs=njobs)(
            joblib.delayed(__inv_op_individual)(i, f, c, loose, depth, outdir) for (i, f, c) in zip(infofs, fwdfs, covfs))

    return saved_files


def __inv_op_individual(infof, fwdf, covf, loose, depth, outdir):

    parts = os.path.basename(infof).split('_')
    num = parts[0]

    fname = f'{outdir}/{parts[0]}_{parts[1]}-inv.fif'

    if os.path.isfile(fname):
        print(f'{fname} already exisits skipping')
        return fname
    try:
        info = mne.io.read_raw_fif(infof, preload=False)
        fwd = mne.read_forward_solution(fwdf)
        cov = mne.read_cov(covf)

        inv = mne.minimum_norm.make_inverse_operator(info.info, fwd, cov, loose=loose, depth=depth)
        mne.minimum_norm.write_inverse_operator(fname, inv)
    except Exception as e:
        print(fname + ' not made for some reason')
        print(e)
    return fname
