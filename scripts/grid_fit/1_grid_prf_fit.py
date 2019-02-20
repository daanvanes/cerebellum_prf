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
# get parameters from command line
########################################################################################

# variable params:
# example call for this script: python 1_grid_prf_fit.py 01 02 23
sub = sys.argv[1] 
ses = sys.argv[2] 
n_jobs = int(sys.argv[3]) 

########################################################################################
# set directories
########################################################################################

# this should point to the directory that contains the tsnr-weighted across run averages:
in_home  = os.path.join('/home','shared','2018','visual','cerebellum_prf','derivatives','pp','zscore')

# this should point to your repo home:
repo_home = os.path.join('/home/vanes/git/cerebellum_prf')

# this should point to the dir where you want the prf results to be saved:
prf_base_dir = os.path.join('/home','shared','2018','visual','cerebellum_prf','derivatives','pp','prf')
if not os.path.isdir(prf_base_dir): os.mkdir(prf_base_dir)

# these automatically follow from the above 
volume_mask_home = os.path.join(repo_home,'resources','volume_masks')
dm_fn = os.path.join(repo_home,'resources','design_matrix.npy')

########################################################################################
# set parameters
########################################################################################

mask_type = 'cerebellum'#gray_matter'#'cerebellum''wang'
fit_type = 'full'# fast

########################################################################################
# other parameters
########################################################################################

# choose filenames:
epi_fn = 'tsnr_weighted_mean_of_resampled_fnirted_smoothed_sgtf_over_runs_ses_%s.nii.gz'%ses
postFix = 'hrf075_nong' # this means no nuissance regression
out_fn = 'new_prf_results_zscore_ses_%s'%ses

# determine fit settings:
animate_dm = False

print('now fitting on subject %s, session %s, n_jobs: %d'%(sub,ses,n_jobs))

TR = 1.5 # in s
hrf_delays = [0.75] # in s

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


# ########################################################################################
# # setup popeye filtered css model
# ########################################################################################

if animate_dm:

    import matplotlib.pyplot as pl
    import matplotlib.animation as animation

    f=pl.figure()
    ims = []
    for t in range(N_TIMEPOINTS):
        pl.title(str(t))
        ims.append([pl.imshow(dm[:,:,t],origin='upper')]) # upper is how popeye wants the y dimension
        pl.axis('off')

    ani = animation.ArtistAnimation(f, ims, interval=1500, blit=True,
                                    repeat_delay=1000)

    fn = '/home/shared/2018/visual/cerebellum_prf/derivatives/figs/dm_animationnew.mp4'
    ani.save(fn)

    f = pl.figure()
    pl.plot(np.sum(np.sum(dm,axis=0),axis=0),'-o')
    pl.savefig('/home/shared/2018/visual/cerebellum_prf/derivatives/figs/dmsum.pdf')

    # # test dm
    pl.figure()
    size = 3
    p1 = model_func.generate_prediction(-10,0,size*sqrt(0.05),0.05,1,0)
    p2 = model_func.generate_prediction(10,0,size*sqrt(0.05),0.05,1,0)
    p3 = model_func.generate_prediction(0,5,size*sqrt(0.05),0.05,1,0) 
    p4 = model_func.generate_prediction(0,-5,size*sqrt(0.05),0.05,1,0) 
    pl.plot(p1,c='b',label='left')
    pl.plot(p2,c='r',label='right')
    pl.plot(p3,c='g',label='top')
    pl.plot(p4,c='orange',label='bottom')

    pl.legend(loc='best')
    pl.savefig('/home/shared/2018/visual/cerebellum_prf/derivatives/figs/test_prediction.pdf')


########################################################################################
# function that will test prediction against all voxels
########################################################################################

def test_param_comb(p,these_data,model_func,N_TIMEPOINTS,n=0.05):

    x,y,ecc,ang,size,hrf_delay = p

    # create a prediction for this parameter combination:
    model_func.hrf_delay = hrf_delay
    prediction = model_func.generate_prediction(x,y,size*np.sqrt(n),n,1,0)
    try:
        this_dm = np.vstack([np.ones_like(prediction),prediction])
        betas, residual, _, _ = np.linalg.lstsq( np.nan_to_num(this_dm.T), np.nan_to_num(these_data.T))
        rsqs = ((1 - residual / (N_TIMEPOINTS * these_data.var(axis=-1)))*10000.).astype('int16')
    except:
        rsqs = np.zeros((these_data.shape[0])).astype('int16')
        betas = np.zeros((2,these_data.shape[0])).astype('int16')

    return rsqs, betas, prediction

def run_model(param_combs,data_to_fit,model_func,N_TIMEPOINTS):

    r2s,betas,predictions = zip(*Parallel(n_jobs=n_jobs,verbose=9)(delayed(test_param_comb)(p,data_to_fit,model_func,N_TIMEPOINTS) for p in param_combs))
    return r2s,betas,predictions

########################################################################################
# now fit
########################################################################################
model_func = CompressiveSpatialSummationModelFiltered(stimulus = stimulus, hrf_model = utils.spm_hrf,
                                                      sg_filter_window_length=120, sg_filter_order=3,
                                                      tr=TR)

# model_func.hrf_delay = hrf_delays[sub]


print 'loading data'
# load data
infn = os.path.join(in_home,'sub-%s'%sub,epi_fn)
input_nii = nb.load(infn)
input_data = np.nan_to_num(input_nii.get_data())
input_shape = input_data.shape

# define output dir
sub_out_dir = os.path.join(prf_base_dir,'sub-%s'%sub)
if not os.path.isdir(sub_out_dir): os.mkdir(sub_out_dir)
opfn = os.path.join(sub_out_dir,'sub-%s_%s.nii.gz'%(sub,out_fn))

# only take non zero voxels
valid_voxels = (np.sum(input_data,axis=-1)!=0)

if mask_type == 'cerebellum':
    # loading cerebellum mask
    mask_fn = os.path.join(volume_mask_home,'cmask.nii')
    mask = nb.load(mask_fn).get_data().astype(bool)
elif mask_type == 'gray_matter':
    mask_fn = os.path.join(volume_mask_home,'avg152T1_gray.hdr')
    mask = np.squeeze((nb.load(mask_fn).get_data()>0.1)) # conservative threshold

# determine mask
mask = mask*valid_voxels
data_to_fit = input_data[mask,:N_TIMEPOINTS] # cut off last TR as this was not in dm
del input_data
del valid_voxels

# determine grid to search:
if fit_type == 'fast':
    # fast fit: resolution half HCP grid # per variable (total grid: 1.152)
    n_eccs = 12
    n_sizes = 12
    n_polars = 32
    max_ecc = 20
    max_size = 16
elif fit_type == 'full':
    # slow fit: resolution same grid size as in Benson HCP (total grid: 10.400)
    n_eccs = 25
    n_sizes = 13
    n_polars = 32
    max_ecc = 20 # HCP: 16, but stim ecc was 8 (here 10)
    max_size = 16 # HCP: 13, but stim ecc was 8 (here 10)

eccs = np.linspace(0.1,1,n_eccs)**3*max_ecc
sizes = np.linspace(0.1,1,n_sizes)**3*max_size
angs = np.linspace(0,np.pi*2,n_polars,endpoint=False)
ncombs = len(eccs)*len(angs)*len(sizes)

# put paramers together
param_combs = []
for ecc in eccs:
    for ang in angs:
        x = ecc * np.cos(ang)
        y = ecc * np.sin(ang)
        for size in sizes:
            for hrf_delay in hrf_delays:
                param_combs.append([x,y,ecc,ang,size,hrf_delay])
param_combs = np.array(param_combs)

# only do the estimation for non 0 voxels:
print 'starting fit on %d param combinations, and on %d voxels'%(ncombs,mask.sum())

# assess different combinations:
r2s,betas,predictions = zip(*Parallel(n_jobs=n_jobs,verbose=9,backend="multiprocessing")(delayed(test_param_comb)(p,data_to_fit,model_func,N_TIMEPOINTS) for p in param_combs))

# determine best fit:
best_fit = np.argmax(r2s,axis=0)
best_r2 = np.max(r2s,axis=0).astype(float)/10000.
del r2s
betas = np.array(betas)
best_betas = np.array([betas[best_fit[v]][:,v] for v in range(data_to_fit.shape[0])])
del betas
best_predictions = np.array(predictions)[best_fit]
del predictions
best_params = np.concatenate([param_combs[best_fit],best_r2[:,np.newaxis],best_betas[:,1][:,np.newaxis]],axis=-1)

# put in volume
params_volume = np.zeros(np.r_[input_shape[:3], np.max([dims[v] for v in dims.keys()])+1])
params_volume[mask,:] = np.array(best_params)

# save
prf_nii = nb.Nifti2Image(params_volume, affine=input_nii.affine, header=input_nii.header)
prf_nii.to_filename(opfn.replace('.nii.gz','%s.nii.gz'%postFix))

# now recreate predictions
print('recreating predictions...')
ps=[]
for v in tqdm(range(mask.sum())):
    pred = best_predictions[v]
    base,amp = best_betas[v]
    ps.append(pred*amp+base)

# put in volume
ps_volume = np.zeros(np.r_[input_shape[:3], N_TIMEPOINTS])
ps_volume[mask,:] = np.array(ps)

# save
prf_nii = nb.Nifti2Image(ps_volume, affine=input_nii.affine, header=input_nii.header)
prf_nii.to_filename(opfn.replace('.nii.gz','%s_predictions.nii.gz'%postFix))

