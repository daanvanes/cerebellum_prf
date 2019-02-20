from __future__ import division
from math import *
import os
import numpy as np
from hrf_estimation.hrf import spmt 
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

########################################################################################
# command line input
########################################################################################

# variable params:
# example call to this script: python 3_create_predictions.py 01 01 10 gray_matter 0.75 myfit 1
sub = sys.argv[1]
ses = sys.argv[2]
n_jobs = int(sys.argv[3])
mask_type = sys.argv[4]
hrf_delay = float(sys.argv[6])
postFix = sys.argv[5]
k = int(sys.argv[7])

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

# choose filenames:
epi_fn = 'tsnr_weighted_mean_of_resampled_fnirted_smoothed_sgtf_over_runs_ses_%s_test_%d.nii.gz'%(ses,k)
out_fn = 'new_prf_results_zscore_ses_%s'%(ses)

# determine fit settings:
animate_dm = False

print('now fitting on subject %s, session %s, n_jobs: %d'%(sub,ses,n_jobs))
# mask_type = 'wang'#gray_matter'#'cerebellum''wang'

TR = 1.5 # in s

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
'n':8,
}

########################################################################################
# load dm
########################################################################################

# load in dm created by experiment capture
dm = np.load(dm_fn)
N_TIMEPOINTS = np.shape(dm)[0]

# put time dimension last for popeye
dm = np.moveaxis(dm,0,2)

# binarize
dm[dm<0.5] = 0
dm[dm>0.5] = 1

# remove fixation point
dm[49,88,:] = 0
dm[49,89,:] = 0
dm[50,88,:] = 0
dm[50,89,:] = 0

#revert y axis
dm = dm[::-1,:,:] # this is how popeye wants y dim (0 point is top of dm)

########################################################################################
# setup popeye filtered css model
########################################################################################

stimulus = VisualStimulus(  stim_arr = dm,
                            viewing_distance = 225, 
                            screen_width =  69.84,
                            scale_factor = 1,
                            tr_length = TR,
                            dtype = np.short)

model_func = CompressiveSpatialSummationModelFiltered(stimulus = stimulus, hrf_model = utils.spm_hrf,
                                                      sg_filter_window_length=120, sg_filter_order=3,
                                                      tr=TR)

print 'loading data'

def cross_predict(d,p,model_func):
    
    x,y,ecc,ang,size,hrf_delay,r2,amp,n = p

    # create a prediction for this parameter combination:
    model_func.hrf_delay = hrf_delay
    try:
        prediction = model_func.generate_prediction(x,y,size*np.sqrt(n),n,amp,0)
        this_dm = np.vstack([np.ones_like(prediction),prediction])
        betas, residual, _, _ = np.linalg.lstsq( np.nan_to_num(this_dm.T), np.nan_to_num(d.T))
        r2 = 1 - residual[0] / (N_TIMEPOINTS * d.var(axis=-1))
        scaled_prediction = betas[0]+betas[1]*prediction
    except:
        r2 = 0.
        scaled_prediction = np.zeros_like(d)

    return r2,scaled_prediction

# load data
infn = os.path.join(in_home,'sub-%s'%sub,epi_fn)
input_nii = nb.load(infn)
input_data = np.nan_to_num(input_nii.get_data())
input_shape = input_data.shape

# load params
sub_out_dir = os.path.join(prf_base_dir,'sub-%s'%sub)
p_ipfn = os.path.join(sub_out_dir,'sub-%s_%s_%s_hrf%.2f_%s_k%d.nii.gz'%(sub,out_fn,postFix,hrf_delay,mask_type,k))
param_nii = nb.load(p_ipfn)
params = np.nan_to_num(param_nii.get_data())

# only take non zero voxels
valid_voxels = (np.sum(input_data,axis=-1)!=0)

# get anatomical roi
if mask_type == 'cerebellum':
    # loading cerebellum mask
    mask_fn = os.path.join(volume_mask_home,'cmask.nii')
    mask = nb.load(mask_fn).get_data().astype(bool)
elif mask_type == 'gray_matter':
    mask_fn = os.path.join(volume_mask_home,'avg152T1_gray.hdr')
    mask = np.squeeze((nb.load(mask_fn).get_data()>0.1)) # conservative threshold

mask *= valid_voxels
data_to_fit = input_data[mask,:N_TIMEPOINTS] # cut off last TR as this was not in dm
these_params = params[mask]
del input_data
del params
del valid_voxels

print 'recreating predictions in %s parallel jobs over %d voxels'%(n_jobs,data_to_fit.shape[0])

r2s, predictions = zip(*Parallel(n_jobs = n_jobs, verbose = 9,backend="multiprocessing")(delayed(cross_predict)
    (data_to_fit[v],these_params[v],model_func) for v in range(data_to_fit.shape[0])))

# put fit results back in volume space
volume = np.zeros(input_nii.shape[:-1])
volume[mask] = r2s

new_image = nb.Nifti1Image(volume,input_nii.affine)
new_image.to_filename(p_ipfn.replace('.nii.gz','_cvr2.nii.gz'))

# put fit results back in volume space
volume = np.zeros(list(input_nii.shape[:3])+[input_nii.shape[3]-1])
volume[mask] = predictions

new_image = nb.Nifti1Image(volume,input_nii.affine)
new_image.to_filename(p_ipfn.replace('.nii.gz','_predictions.nii.gz'))


