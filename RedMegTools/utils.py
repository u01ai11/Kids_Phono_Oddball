import mne
import os
import numpy as np
import joblib

def check_ids(rawdir, fsdir, mnedir):
    """
    Takes three directories and checks for necccessary files for analysis

    :param rawdir:
        The directory where some raw files are, IDs are extracted from these
    :param fsdir:
        The directory for freesurfer files, we check here for recon-all results
    :param mnedir:
        The directory where all MNE files are being saved
    :return resmatrix:
        The results in a type x part_no matrix. Cell a '' if not present

    """

    rawflist = [f for f in os.listdir(rawdir) if os.path.isfile(os.path.join(rawdir, f))]
    fsdirlist = [f for f in os.listdir(fsdir) if os.path.isdir(os.path.join(fsdir, f))]
    mnelist = [f for f in os.listdir(mnedir) if os.path.isfile(os.path.join(mnedir, f))]
    subnames_only = list(set([x.split('_')[0] for x in rawflist]))  # get a unique list of IDs

    fs_match = []
    trans_match = []
    bemm_match = []
    bems_match = []
    cov_match = []
    fwd_match = []

    for sub in subnames_only:

        # check for freesurfer folder
        matching = [s for s in fsdirlist if sub in s]# get matching item(s)
        if len(matching) > 0: # if there are any
            matching2 = [s for s in matching if 'scaled' in s]
            if len(matching2) > 0:
                fs_match.append(matching2)
            else:
                fs_match.append('')
        else:
            fs_match.append('')

        # check for trans
        matching = [s for s in mnelist if sub in s]
        if len(matching) > 0: # if there are any
            matching2 = [s for s in matching if 'trans.fif' in s]
            if len(matching2) > 0:
                trans_match.append(matching2)
            else:
                trans_match.append('')
        else:
            trans_match.append('')

        # check for BEM model
        matching = [s for s in mnelist if sub in s]
        if len(matching) > 0: # if there are any
            matching2 = [s for s in matching if 'bem.fif' in s]
            if len(matching2) > 0:
                bemm_match.append(matching2)
            else:
                bemm_match.append('')
        else:
            bemm_match.append('')


        # check for BEM solution
        matching = [s for s in mnelist if sub in s]
        if len(matching) > 0: # if there are any
            matching2 = [s for s in matching if 'bem-sol.fif' in s]
            if len(matching2) > 0:
                bems_match.append(matching2)
            else:
                bems_match.append('')
        else:
            bems_match.append('')

    return [fs_match,
            trans_match,
            bemm_match,
            bems_match]

def find_fwd_files(ids, mnedir, megdir):
    """
    Finds the files needed for running making a forward solution in MNE
    :param ids:
        list of ids to look for files
    :param mnedir:
        directory containing those files for MNE sourcespace
    :param megdir:
        directory containing meg files needed for info
    :return:
        array with:
        list of meg files
        list of trans files
        list of sourcespace
        list of bem solution files
    """

    megfiles = []
    transfiles = []
    sourcefiles = []
    bemfiles = []

    mnefs = os.listdir(mnedir)
    megfs = os.listdir(megdir)
    for id in ids:
        allidf = [f for f in mnefs if id in f]
        num = id.split('_')[0] # just get number
        allmegf =[ f for f in megfs if num in f]

        # append a meg file if there, else an empty string
        if len(allmegf) > 0:
            megfiles.append(allmegf[0])
        else:
            megfiles.append('')

        # look for trans
        transf = [f for f in allidf if 'trans.fif' in f]
        if len(transf) > 0:
            transfiles.append(transf[0])
        else:
            transfiles.append('')

        # look for sourcespaces
        srcf = [f for f in allidf if 'src.fif' in f]
        if len(srcf) > 0:
            sourcefiles.append(srcf[0])
        else:
            sourcefiles.append('')

        # look for bemsol files
        bemsf = [f for f in allidf if 'bem-sol.fif' in f]
        if len(bemsf) > 0:
            bemfiles.append(bemsf[0])
        else:
            bemfiles.append('')

    return (megfiles, transfiles, sourcefiles, bemfiles)



