import mne
import os
import numpy as np
import joblib


def recon_all_multiple(sublist, struct_dir, fs_sub_dir, fs_call, njobs, cbu_clust)

    """
    :param sublist: 
        A list of subjects for source recon 
    :param struct_dir: 
        A directory containing MRI scans with name format:
        subname + T1w.nii.gz 
    :param fs_sub_dir: 
        The Freesurfer subject directory 
    :param fs_call: 
        The call for freesurfer, specific to the version we want to use 
    :param njobs: 
        If not CBU cluster, we will commit jobs on the local machine.
        Relates to no, of parallel jobs 
    :param cbu_clust: 
        If true this will submit jobs to the CBU cluster queue using 'qsub'
    :return: 
    """
    if not cbu_clust:
        # set up
        # TODO: make bash version, this will only work on tcsh terminal
        # check if dir exists, make if not
        if os.path.isdir(fs_sub_dir):
            os.system(f"tcsh -c '{fs_call} && setenv SUBJECTS_DIR {fs_sub_dir}'")
        else:
            os.system(f"tcsh -c '{fs_call} && mkdir {fs_sub_dir} && setenv SUBJECTS_DIR {fs_sub_dir}'")

        saved_files = []

        if njobs == 1:
            for i in range(len(sublist)):
                savedfile = __recon_all_individual(os.path.join(sublist[i], struct_dir, fs_sub_dir))
                saved_files.append(savedfile)
        if njobs > 1:

            saved_files = joblib.Parallel(n_jobs =njobs)(
                joblib.delayed(recon_all_individua)(os.path.join(thisS, struct_dir, fs_sub_dir)) for thisS in sublist)

        return saved_files

    else:
        saved_files = []
        # We are using CBU cluster so construct qstat jobs
        for i in range(len(sublist)):
            savedfile = __recon_all_qstat(sublist[i], struct_dir, fs_sub_dir)
            saved_files.append(savedfile)

def __recon_all_individual(sub, struct_dir, fs_sub_dir):
    """
    Private function for recon using freesurfer in tcsh shell

    :param sub:
        Subject name/number
    :param struct_dir:
        Where to find that subjects T1 weighted structural MRI scan
    :param fs_sub_dir:
        The subject dir for fressurfer, only used to return the
    :return this_sub_dir:
        The directory where freesurfer is storring it's recon files
    """
    T1_name = sub + '_T1w.nii.gz'

    if os.path.isfile(struct_dir + '/' + T1_name):
        os.system(f"tcsh -c 'recon-all -i {struct_dir}/{T1_name} -s {sub} -all -parallel'")
    else:
        print('no T1 found for ' + sub)

    this_sub_dir = f'{fs_sub_dir}/{sub}'
    return this_sub_dir

def __recon_all_qstat(sub, struct_dir, fs_sub_dir):
    """
    Private function for submitting source-recon freesurfer commands to CBU's cluster
    :param sub:
        Subject name/number
    :param struct_dir:
        Where to find that subjects T1 weighted structural MRI scan
    :param fs_sub_dir:
        The subject dir for fressurfer, only used to return the
    :return this_sub_dir:
        The directory where freesurfer is storring it's recon files

    """
    T1_name = sub + '_T1w.nii.gz'

    # construct tcsh command
    qsub_com =\
        f"""
        #!/bin/tcsh
        freesurfer_6.0.0 
        setenv SUBJECTS_DIR {fs_sub_dir}
        recon-all -i {struct_dir}/{T1_name} -s {sub} -parallel -openmp 8
        """
    #save to a csh script
    with open (f'{sub}.csh', "w") as c_file:
        c_file.write(qsub_com)

    # construct the qsub command and execute
    os.system(f"tcsh -c 'sbatch -job-name=reco_{sub} -mincpus=8 -t=2-1:10 {sub}.csh'")

    # submit
    this_sub_dir = f'{fs_sub_dir}/{sub}'
    return this_sub_dir


