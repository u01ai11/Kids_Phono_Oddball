#!/usr/bin/python

# vim: set expandtab ts=4 sw=4:

# Adapted from A.J. Quinn glm
import sys
sys.path.insert(0, '/home/ai05/Downloads/glm')
import numpy as np
from glmtools import fit,regressors
import pickle
import os
import time
import _pickle

def permute_glm( glmdes, data, nperms=5000, stat='cope',nomax_axis=None,
        temporal_varcope_smoothing=None):
    """
    Permute rows of design matrix to generate null distributions
    """

    f = fit.OLSModel( glmdes, data )
    dims = tuple(np.arange(1,f.copes.ndim))

    nulls = np.zeros( (glmdes.num_contrasts,nperms, data.data.shape[1], data.data.shape[2]))

    if stat == 'cope':
        nulls[:,0,...] = f.copes
    elif stat == 'tstat':
        f.time_dim = 2
        tstats = f.get_tstats(temporal_varcope_smoothing=temporal_varcope_smoothing)
        nulls[:,0,...] = tstats


    x = glmdes.design_matrix.copy()
    from copy import deepcopy
    g = deepcopy( glmdes )

    # Indices of regressors of interest for each contrast
    cinds = [ np.where( glmdes.contrasts[ii,:] != 0.)[0] for ii in range(glmdes.num_contrasts) ]

    for jj in range( glmdes.num_contrasts):

        ctype = glmdes.contrast_list[jj].ctype
        rtype = [ glmdes.regressor_list[ii].rtype for ii in cinds[jj] ]
        rtype = np.unique(rtype)
        if len(rtype) > 1:
            raise ValueError('Contrast is mixing multiple regressor types')
        else:
            rtype = rtype[0]

        mode = None
        if rtype == 'Categorical':
            if ctype == 'Differential':
                mode = 'row-shuffle'
            else:
                mode = 'sign-flip'
        elif rtype == 'Parametric':
            mode = 'row-shuffle'
        elif rtype == 'Continous':
            mode = 'row-shuffle'

        if mode is None:
            raise ValueError('unable to determine mode')

        print('Permuting {0} by {1}'.format(glmdes.contrast_list[jj],mode))
        for ii in range(1,nperms):

            g.design_matrix = apply_permutation( x.copy(), cinds[jj], mode )

            f = fit.OLSModel( g, data )
            if stat=='cope':
                nulls[jj,ii] = np.abs(f.copes[jj,...])
            elif stat=='tstat':
                f.time_dim = 2
                tstats = f.get_tstats(temporal_varcope_smoothing=temporal_varcope_smoothing)
                nulls[jj,ii] = tstats[jj,...]
            else:
                print('stat not recognised: please use stat=\'cope\' or stat=\'tstat\'')

    return nulls

def permute_glm_cluster( glmdes, data, scriptdir, pythondir, filesdir, nperms=5000, stat='cope',
                         nomax_axis=None, temporal_varcope_smoothing=None):
    """
    :param glmdes: Design matrix
    :param data: Data
    :param nperms: number of times to permute
    :param stat: Statistic to permute
    :param nomax_axis: max axis for distributions
    :param temporal_varcope_smoothing: if tstat, timewindow to smotth
    :param scriptdir: directory to save out cluster scripts to
    :param pythondir: python directory
    :param filesdir: directory to save output files to
    :return:
    """

    #scriptdir = '/imaging/ai05/phono_oddball/cluster_scripts'
    #filesdir = '/imaging/ai05/phono_oddball/cluster_files'

    f = fit.OLSModel( glmdes, data )

    nulls = np.zeros( (glmdes.num_contrasts,nperms, data.data.shape[1], data.data.shape[2]))

    if stat == 'cope':
        nulls[:,0,...] = f.copes
    elif stat == 'tstat':
        f.time_dim = 2
        tstats = f.get_tstats(temporal_varcope_smoothing=temporal_varcope_smoothing)
        nulls[:,0,...] = tstats


    x = glmdes.design_matrix.copy()
    from copy import deepcopy
    g = deepcopy( glmdes )

    # Indices of regressors of interest for each contrast
    cinds = [ np.where( glmdes.contrasts[ii,:] != 0.)[0] for ii in range(glmdes.num_contrasts) ]


    for jj in range( glmdes.num_contrasts):

        ctype = glmdes.contrast_list[jj].ctype
        rtype = [ glmdes.regressor_list[ii].rtype for ii in cinds[jj] ]
        rtype = np.unique(rtype)
        if len(rtype) > 1:
            raise ValueError('Contrast is mixing multiple regressor types')
        else:
            rtype = rtype[0]

        mode = None
        if rtype == 'Categorical':
            if ctype == 'Differential':
                mode = 'row-shuffle'
            else:
                mode = 'sign-flip'
        elif rtype == 'Parametric':
            mode = 'row-shuffle'
        elif rtype == 'Continous':
            mode = 'row-shuffle'

        if mode is None:
            raise ValueError('unable to determine mode')
        # we now need to save g, x, glemdes, cinds and data
        saveobject = (g, x, cinds, glmdes, data)
        os.makedirs(os.path.dirname(filesdir), exist_ok=True)
        with open(f'{filesdir}/temp_info_{jj}.pkl', "wb") as f:
            _pickle.dump(saveobject, f)


        # make function for this contrast, taking only ii as input argument
        pycom = f"""
import sys
sys.path.insert(0, '/home/ai05/Downloads/glm')
import numpy as np
from glmtools import fit,regressors
import pickle
import _pickle


def apply_permutation( X, cinds, mode ):

    if mode == 'sign-flip':
        I = np.random.permutation(np.tile( [1,-1], int(X.shape[0]/2)))
        X[:,cinds] = X[:,cinds] * I[:,None]
    elif mode == 'row-shuffle':
        I = np.random.permutation(X.shape[0])
        ix = np.ix_(I,cinds) # Can't apply indexing to both dims
        X[:,cinds] = X[ix]

    return X

# prefefined stuff 

ii = sys.argv[1] # input number from SLURM array 
jj = {jj}
mode = '{mode}'
stat = '{stat}'

#data from files 
with open('{filesdir}/temp_info_{jj}.pkl', "rb") as f:
    saveobject = _pickle.load(f)
g, x, cinds, glmdes, data = saveobject


g.design_matrix = apply_permutation( x.copy(), cinds[jj], mode )
f = fit.OLSModel( g, data )

if stat =='cope':
    out = np.abs(f.copes[jj,...])
elif stat=='tstat':
    f.time_dim = 2
    tstats = f.get_tstats(temporal_varcope_smoothing=temporal_varcope_smoothing)
    out  = tstats[jj,...]
else:
    print('stat not recognised: please use stat=cope or stat=tstat')

# save array 
print('saving file')
np.save('{filesdir}/{jj}_'+str(ii)+'.npy', out)
print('file save complete')
"""
        print(pycom, file=open(f'{scriptdir}/batch_perm_{jj}.py', 'w'))
        # make required files
        print('Permuting {0} by {1}'.format(glmdes.contrast_list[jj],mode))
        # for ii in range(1,nperms):
        #     # copy and rename resources
        #     os.system(f'cp {filesdir}/temp_info.pkl {filesdir}/temp_info_{jj}{ii}.pkl')
        #     # template script for looping over
        #     print(ii)


        # construct sh file
        shf = f"""#!/bin/bash
#SBATCH -t 0-1:00
#SBATCH --job-name=alex_perm_465
#SBATCH --mincpus=1
#SBATCH --out={jj}_%j.out
#SBATCH --requeue 
#SBATCH -a 1-{nperms}
{pythondir} {scriptdir}/batch_perm_{jj}.py $SLURM_ARRAY_TASK_ID
              """
        # save to directory
        print(shf, file=open(f'{scriptdir}/batch_perm_{jj}.csh', 'w'))

        # execute this on the cluster
        os.system(f'sbatch {scriptdir}/batch_perm_{jj}.csh')


    # wait until all permutations are done
    starttime = time.time()

    #update every second and continue when finished
    completed = False
    while completed == False:
        queued = len(os.popen('squeue -n alex_perm_465').read().split('\n'))-2
        perc_done = 1-(queued/((nperms-1)*glmdes.num_contrasts))

        sys.stdout.write("\rClustering %i percent" % round(perc_done*100,2))
        sys.stdout.flush()
        if perc_done == 1.0:
            completed = True
        time.sleep(1.0 - ((time.time() - starttime) % 1.0))

    # Now we have to load all the files
    for jj in range(glmdes.num_contrasts):
        for ii in range(1,nperms):
            #load
            infile = np.load(f'{filesdir}/{jj}_{ii}.npy')
            nulls[jj, ii] = infile
            os.system(f'rm {filesdir}/{jj}_{ii}.npy') # delete

    #return this
    return nulls



def apply_permutation( X, cinds, mode ):

    if mode == 'sign-flip':
        I = np.random.permutation(np.tile( [1,-1], int(X.shape[0]/2)))
        X[:,cinds] = X[:,cinds] * I[:,None]
    elif mode == 'row-shuffle':
        I = np.random.permutation(X.shape[0])
        ix = np.ix_(I,cinds) # Can't apply indexing to both dims
        X[:,cinds] = X[ix]

    return X
