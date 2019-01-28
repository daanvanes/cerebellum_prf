from __future__ import division
from math import *
import os
import numpy as np
from hrf_estimation.hrf import spmt  # , dspmt, ddspmt
from scipy.signal import savgol_filter, fftconvolve, resample
import nibabel as nb
from sklearn.linear_model import Ridge
from IPython import embed as shell 
import copy
from tqdm import tqdm
from joblib import Parallel, delayed

# import popeye 
from popeye.visual_stimulus import VisualStimulus
from css_sg import CompressiveSpatialSummationModelFiltered
import popeye.utilities as utils
import popeye.css as css
import sys

########################################################################################
# set parameters
########################################################################################


# variable params:
sub = sys.argv[1]
ses = sys.argv[2]

n_folds = 10

# choose filenames:
epi_fn = 'tsnr_weighted_mean_of_resampled_fnirted_smoothed_sgtf_over_runs_ses_%s.nii.gz'%ses
# epi_fn = 'tsnr_weighted_mean_zscore_over_runs_ses_%s.nii.gz'%ses#'mean_zscore_over_all_runs_MNI.nii.gz'
postFix = 'hrf075_nong'
out_fn = 'new_prf_results_zscore_ses_%s'%ses

# setup dirs
in_home  = os.path.join('/home','shared','2018','visual','cerebellum_prf','derivatives','pp','zscore')
prf_base_dir = os.path.join('/home','shared','2018','visual','cerebellum_prf','derivatives','pp','prf')

# save dims:
dims = {
'x':0,
'y':1,
'ecc':2,
'ang':3,
'size':4,
'hrf_delay':5,
'r2':6,
'amp':7,
}

######################################
# evaluate fits
######################################
all_r2s = []
for k in range(n_folds):
    
    sub_out_dir = os.path.join(prf_base_dir,'sub-%s'%sub)

    this_out_fn = os.path.join(sub_out_dir,'sub-%s_%s%s_k%d_cvr2.nii.gz'%(sub,out_fn,postFix,k))
    # if not os.path.isfile(this_out_fn):
    # load test data
    print 'loading test data of fold %d'%k
    infn = os.path.join(in_home,'sub-%s'%sub,epi_fn).replace('.nii.gz','_test_%d.nii.gz'%k)
    input_nii = nb.load(infn)
    input_data = np.nan_to_num(input_nii.get_data())
    input_shape = input_data.shape

    # load predictions
    print 'loading predictions of fold %d'%k    
    pred_fn = os.path.join(sub_out_dir,'sub-%s_%s.nii.gz'%(sub,out_fn)).replace('.nii.gz','%s_k%d_predictions.nii.gz'%(postFix,k))
    pred_nii = nb.load(pred_fn)
    pred_data = np.nan_to_num(pred_nii.get_data())

    # assess different combinations:
    print 'computing cv r2 for fold %d'%k    
    residual = np.nan_to_num(np.sum((input_data[:,:,:,:-1]-pred_data)**2,axis=-1))
    r2s = (1 - residual / (np.shape(pred_data)[-1] * input_data.var(axis=-1)))
    r2s[r2s==1] = 0
    all_r2s.append(r2s)

    # save
    print 'saving cvr2 fold %d'%k    
    prf_nii = nb.Nifti2Image(r2s, affine=input_nii.affine, header=input_nii.header)
    prf_nii.to_filename(out_fn)
    # else:
    #     all_r2s.append(nb.load(this_out_fn).get_data())

# # avg r2
# print 'saving avg cvr2 over folds'
avg_r2 = np.mean(all_r2s,axis=0)
# prf_nii = nb.Nifti2Image(avg_r2, affine=input_nii.affine, header=input_nii.header)
# prf_nii.to_filename(pred_fn.replace('k%d_predictions'%k,'cvr2'))

######################################
# create weighted avg of parameters
######################################

#####################################
all_params = []
for k in range(n_folds):

    print 'loading data'

    # load params
    sub_out_dir = os.path.join(prf_base_dir,'sub-%s'%sub)
    param_fn = os.path.join(sub_out_dir,'sub-%s_%s.nii.gz'%(sub,out_fn)).replace('.nii.gz','%s_k%d.nii.gz'%(postFix,k))
    param_nii = nb.load(param_fn)
    params = np.nan_to_num(param_nii.get_data())
    all_params.append(params)
all_params = np.array(all_params)

avg_params = np.zeros(np.shape(all_params)[1:])
all_r2s = np.array(all_r2s)
all_r2s[all_r2s==0] = 1e5
for p in dims.keys():
    if p == 'ang':
        avgx = np.average(all_params[:,:,:,:,dims['x']],weights=all_r2s,axis=0)
        avgy = np.average(all_params[:,:,:,:,dims['y']],weights=all_r2s,axis=0)
        avg = np.arctan2(avgy, avgx)
        avg[avg<0] = 2*np.pi+avg[avg<0] # rescale so runs from 0-2pi
    elif p == 'ecc':
        avgx = np.average(all_params[:,:,:,:,dims['x']],weights=all_r2s,axis=0)
        avgy = np.average(all_params[:,:,:,:,dims['y']],weights=all_r2s,axis=0)
        avg = np.linalg.norm(np.array([avgx,avgy]),axis=0)
    else:
        avg = np.average(all_params[:,:,:,:,dims[p]],weights=all_r2s,axis=0)
    avg_params[:,:,:,dims[p]] = avg

avg_params[:,:,:,dims['r2']] = avg_r2

prf_nii = nb.Nifti2Image(avg_params, affine=param_nii.affine, header=param_nii.header)
prf_nii.to_filename(param_fn.replace('_k%d'%k,'cv'))





