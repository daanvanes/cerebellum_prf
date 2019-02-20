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
import socket 

# import popeye 
from popeye.visual_stimulus import VisualStimulus
from css_sg import CompressiveSpatialSummationModelFiltered
import popeye.utilities as utils
import popeye.css as css
import sys

sys.path.append('/home/vanes/cerebellum_prf/scripts/')
from create_mask import create_mask

########################################################################################
# command line inputs
########################################################################################

# example call to this script: python 4_evaluate_popeye_cv.py 01 01 gray_matter 0.75 myfit 
sub = sys.argv[1]
ses = sys.argv[2]
mask_type = sys.argv[3]
hrf_delay = float(sys.argv[4])
postFix = sys.argv[5]
ks = np.range(10)

########################################################################################
# set dirs
########################################################################################

if socket.gethostname() == 'aeneas':
    # set these
    # dir where input timecourses are located:
    in_home  = os.path.join('/home','shared','2018','visual','cerebellum_prf','derivatives','pp','zscore')
    # where s repo stored:
    repo_dir = os.path.join('/home','vanes','git','cerebellum_prf')
    # where should prf results be saved:
    prf_base_dir = os.path.join('/home','shared','2018','visual','cerebellum_prf','derivatives','pp','prf')

else:
    # set these
    # dir where input timecourses are located:
    in_home  = os.path.join('/projects','0','pqsh283','cerprf','zscore')
    # where is repo cloned:
    repo_dir = os.path.join('/home','vanes','git','cerebellum_prf')
    # where should prf results be saved:
    prf_base_dir = os.path.join('/projects','0','pqsh283','cerprf','prf')

# these follow from above:
volume_mask_home = os.path.join(repo_dir,'resources','volume_masks')
if not os.path.isdir(prf_base_dir): os.mkdir(prf_base_dir)
dm_fn = os.path.join(repo_dir,'resources','design_matrix.npy')

########################################################################################
# set parameters
########################################################################################

epi_fn = 'tsnr_weighted_mean_of_resampled_fnirted_smoothed_sgtf_over_runs_ses_%s.nii.gz'%ses
out_fn = 'new_prf_results_zscore_ses_%s'%(ses)

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
'n':8}

######################################
# get r2s
######################################
sub_out_dir = os.path.join(prf_base_dir,'sub-%s'%sub)

all_r2s = []
for k in ks:

    r2_fn = os.path.join(sub_out_dir,'sub-%s_%s.nii.gz'%(sub,out_fn)).replace('.nii.gz','_%s_hrf%.2f_%s_k%d_cvr2.nii.gz'%(postFix,hrf_delay,mask_type,k))
    print 'loading %s'%r2_fn
    r2_nii = nb.load(r2_fn)
    all_r2s.append(r2_nii.get_data())

avg_r2 = np.mean(all_r2s,axis=0)

######################################
# create weighted avg of parameters
######################################

#####################################
all_params = []
for k in ks:

    # load params
    sub_out_dir = os.path.join(prf_base_dir,'sub-%s'%sub)
    param_fn = os.path.join(sub_out_dir,'sub-%s_%s.nii.gz'%(sub,out_fn)).replace('.nii.gz','_%s_hrf%.2f_%s_k%d.nii.gz'%(postFix,hrf_delay,mask_type,k))
    print 'loading %s'%param_fn
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
        print np.nanmax(avg)
    elif p == 'ecc':
        avgx = np.average(all_params[:,:,:,:,dims['x']],weights=all_r2s,axis=0)
        avgy = np.average(all_params[:,:,:,:,dims['y']],weights=all_r2s,axis=0)
        avg = np.linalg.norm(np.array([avgx,avgy]),axis=0)
    else:
        avg = np.average(all_params[:,:,:,:,dims[p]],weights=all_r2s,axis=0)
    avg_params[:,:,:,dims[p]] = avg

avg_params[:,:,:,dims['r2']] = avg_r2

prf_nii = nb.Nifti2Image(avg_params, affine=param_nii.affine, header=param_nii.header)
prf_nii.to_filename(param_fn.replace('_k%d'%k,'_cv'))

######################################
# create weighted avg of parameters
######################################

#####################################
all_preds = []
for k in ks:

    # load params
    sub_out_dir = os.path.join(prf_base_dir,'sub-%s'%sub)
    pred_fn = os.path.join(sub_out_dir,'sub-%s_%s.nii.gz'%(sub,out_fn)).replace('.nii.gz','_%s_hrf%.2f_%s_k%d_predictions.nii.gz'%(postFix,hrf_delay,mask_type,k))
    print 'loading %s'%pred_fn
    pred_nii = nb.load(pred_fn)
    preds = np.nan_to_num(pred_nii.get_data())
    all_preds.append(preds)

all_preds = np.array(all_preds)
all_r2s = np.nan_to_num(all_r2s)
all_r2s[all_r2s==0] = 1e5

print 'computing across-run prediction average'
avg_pred = np.average(all_preds,weights=np.tile(all_r2s[:,:,:,:,np.newaxis],(1,1,1,1,119)),axis=0)

print 'saving'
prf_nii = nb.Nifti2Image(avg_pred, affine=pred_nii.affine, header=pred_nii.header)
prf_nii.to_filename(pred_fn.replace('_k%d'%k,''))



