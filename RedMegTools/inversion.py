import mne
import os
import numpy as np
import joblib


def fwd_solution_multiple(raw_list, trans_list, src_list, bems_list, outdir, n_jobs):


    # set up

    # check if dir exists, make if not

    saved_files = []

    if n_jobs1 == 1:
        for i in range(len(raw_list)):
            savedfile = __setup_src_individual(sublist[i], fs_sub_dir, outdir, spacing, surface, n_jobs2)
            saved_files.append(savedfile)
    if n_jobs1 > 1:
        saved_files = joblib.Parallel(n_jobs =n_jobs)(
            joblib.delayed(__setup_src_individual)(thisS, fs_sub_dir, outdir, spacing, surface, n_jobs2) for thisS in sublist)

    return saved_files

def __fwd_individual(raw, trans, src, bemsol, outdir):

    f_only = os.path.basename(raw).split('_')  # get filename parts seperated by _
    num = f_only[0]

    rawf = mne.io.read_raw_fif(raw, preload=False)
    srcf = mne.read_source_spaces(src)
    bemsolf = mne.read_bem_solution(bemsol)
    fwd = mne.make_forward_solution(rawf.info, trans, srcf, bemsolf)
    mne.write_forward_solution(f'{outdir}/{num}-fwd.fif', fwd)

    return this_sub_dir