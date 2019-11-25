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

def permute_glm_cluster( glmdes, data, nperms=5000, stat='cope',nomax_axis=None,
        temporal_varcope_smoothing=None, scriptdir, pythondir, filesdir):
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
        with open(f'{filesdir}/temp_info.pkl', "wb") as f:
            pickle.dump(saveobject, f)

        print('Permuting {0} by {1}'.format(glmdes.contrast_list[jj],mode))
        for ii in range(1,nperms):

            # template script for looping over
            pycom = f"""
import sys
sys.path.insert(0, '/home/ai05/Downloads/glm')
import numpy as np
from glmtools import fit,regressors
import pickle

with open(f'{filesdir}/temp_info.pkl', "rb") as f:
    saveobject = pickle.load(f)

g, x, cinds, glmdes, data = saveobject

def apply_permutation( X, cinds, mode ):

    if mode == 'sign-flip':
        I = np.random.permutation(np.tile( [1,-1], int(X.shape[0]/2)))
        X[:,cinds] = X[:,cinds] * I[:,None]
    elif mode == 'row-shuffle':
        I = np.random.permutation(X.shape[0])
        ix = np.ix_(I,cinds) # Can't apply indexing to both dims
        X[:,cinds] = X[ix]

    return X

jj = {jj}
mode = {mode}
stat = {stat}

g.design_matrix = apply_permutation( x.copy(), cinds[jj], mode )
f = fit.OLSModel( g, data )

if stat =='cope':
    out = np.abs(f.copes[jj,...])
elif stat=='tstat':
    f.time_dim = 2
    tstats = f.get_tstats(temporal_varcope_smoothing=temporal_varcope_smoothing)
    out  = tstats[jj,...]
else:
    print('stat not recognised: please use stat=\'cope\' or stat=\'tstat\'')

# save array 
np.save('{filesdir}/{jj}_{ii}.npy', out)
"""

            # submit cluster job
            # save to file
            print(pycom, file=open(f'{scriptdir}/batch_perm.py', 'w'))

            # construct csh file
            tcshf = f"""#!/bin/tcsh
                 {pythondir} {scriptdir}/batch_perm.py
                         """
            # save to directory
            print(tcshf, file=open(f'{scriptdir}/batch_perm.csh', 'w'))

            # execute this on the cluster
            os.system(f'sbatch --job-name=alex_perm_465 --mincpus=1 -t 0-8:00 {scriptdir}/batch_perm.csh')

            os.system()


    # wait until all permutations are done
    starttime = time.time()

    #update every second and continue when finished
    completed = False
    while ~completed:
        queued = len(os.popen('squeue -n alex_perm_465').read().split('\n'))-2
        perc_done = 1-(queued/((nperms-1) * glmdes.num_contrasts))

        if perc_done == 1:
            completed = True
        time.sleep(1.0 - ((time.time() - starttime) % 1.0))

    # Now we have to load all the files
    for jj in range(glmdes.num_contrasts):
        for ii in range(1,nperms):
            #load
            infile = np.load(f'{filesdir}/{jj}_{ii}.npy')
            nulls[jj, ii] = infile

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
