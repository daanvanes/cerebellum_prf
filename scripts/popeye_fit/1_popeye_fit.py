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
# set parameters
########################################################################################

# variable params:
sub = sys.argv[1]
ses = sys.argv[2]
n_jobs = int(sys.argv[3])
# n_folds = 10
sl = int(sys.argv[4])
mask_type = sys.argv[5]
hrf_delay = float(sys.argv[7])
postFix = sys.argv[6]+'_hrf%.2f'%hrf_delay
k = int(sys.argv[8])
# choose filenames:
epi_fn = 'tsnr_weighted_mean_of_resampled_fnirted_smoothed_sgtf_over_runs_ses_%s_train_%d.nii.gz'%(ses,k)
# epi_fn = 'tsnr_weighted_mean_zscore_over_runs_ses_%s.nii.gz'%ses#'mean_zscore_over_all_runs_MNI.nii.gz'
# postFix = 'popeye'
out_fn = 'new_prf_results_zscore_ses_%s'%(ses)

# determine fit settings:
animate_dm = False

print('now fitting on subject %s, session %s, n_jobs: %d'%(sub,ses,n_jobs))
# mask_type = 'wang'#gray_matter'#'cerebellum''wang'

TR = 1.5 # in s
# hrf_delay = 0#np.linspace(-TR,TR,5)
# hrf_delays = np.linspace(-TR*2,TR*2,9)

# setup dirs
if socket.gethostname() == 'aeneas':
    in_home  = os.path.join('/home','shared','2018','visual','cerebellum_prf','derivatives','pp','zscore')
    volume_mask_home = os.path.join('/home','shared','2018','visual','cerebellum_prf','resources','volume_masks')
    mni_home  = os.path.join('/home','vanes','bin','fsl','data','standard')
    prf_base_dir = os.path.join('/home','shared','2018','visual','cerebellum_prf','derivatives','pp','prf')
    if not os.path.isdir(prf_base_dir): os.mkdir(prf_base_dir)
    dm_fn = os.path.join('/home','shared','2018','visual','cerebellum_prf','resources','design_matrix.npy')
else:
    home = os.path.join('/projects','0','pqsh283','cerprf')
    in_home  = os.path.join(home,'zscore')
    volume_mask_home = os.path.join(home,'resources','volume_masks')
    # mni_home  = os.path.join('/home','vanes','bin','fsl','data','standard')
    prf_base_dir = os.path.join(home,'prf')
    dm_fn = os.path.join(home,'resources','design_matrix.npy')  

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

model_func.hrf_delay = hrf_delay

print 'loading data'

# load data
# infn = os.path.join(in_home,'sub-%s'%sub,epi_fn).replace('.nii.gz','_train_%d.nii.gz'%k)
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
sl_mask = np.zeros(input_data.shape[:-1]).astype(bool)

# get anatomical roi
mask = create_mask(mask_type)

# mask by slice number
sls = np.arange(sl*7,(sl+1)*7)
sl_mask[:,:,sls] = mask[:,:,sls]
sl_mask *= valid_voxels
data_to_fit = input_data[sl_mask,:N_TIMEPOINTS] # cut off last TR as this was not in dm
del input_data
del valid_voxels

if sl_mask.sum()==0:
    print 'no voxels in this slice'
else:
    ### FIT
    ## define search grids
    # these define min and max of the edge of the initial brute-force search.
    grid_size = 4               # amount of values between min and max
    x_grid = (-10,10)             # x
    y_grid = (-5,5)         # y
    s_grid = (1,10)              # size
    n_grid = (0.05, 1)       # nonlinearity

    # define search bounds
    # these define the boundaries of the final gradient-descent search.
    x_bound = (-50, 50)           # x 
    y_bound = (-50, 50)           # y 
    s_bound = (0.001, 100)          # size 
    beta_bound = (-25,25)        # amplitude
    n_bound = (0.001, 1.5)          # nonlinearity
    base_bound = (-10, 10)        # baseline

    # order of css estimate parameters:
    css_grids = (x_grid, y_grid, s_grid, n_grid)
    css_bounds = (x_bound, y_bound, s_bound, n_bound, beta_bound, base_bound)

    print 'starting fit in %s parallel jobs over %d voxels'%(n_jobs,data_to_fit.shape[0])

    res = Parallel(n_jobs = n_jobs, verbose = 9,backend="multiprocessing")(delayed(css.CompressiveSpatialSummationFit)
    (model_func,vox,css_grids,css_bounds,voxel_index=(0,0,vi),Ns=grid_size,verbose=0)
            for vi,vox in enumerate(data_to_fit))     

    # first get the estimates to save
    data_array = []
    for vi in range(len(res)):
        ecc = np.linalg.norm(res[vi].estimate[:2])
        size = res[vi].sigma_size
        theta = res[vi].theta
        temp = [res[vi].estimate[0],res[vi].estimate[1],ecc,theta,size,hrf_delay,res[vi].rsquared,res[vi].beta,res[vi].n]
        # temp = list(res[vi].estimate) + [ecc] + [res[vi].rsquared] +[size] + [theta]
        data_array.append(temp)
    data_array = np.array(data_array)

    # put fit results back in volume space
    volume = np.zeros(list(input_nii.shape[:-1])+[np.shape(data_array)[1]])
    volume[sl_mask] = data_array

    new_image = nb.Nifti1Image(volume,input_nii.affine)
    new_image.to_filename(opfn.replace('.nii.gz','_%s_%s_sl%d_k%d.nii.gz'%(postFix,mask_type,sl,k)))

