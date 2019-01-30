import nibabel as nb
import os
import numpy as np
import copy
import sys 
from sklearn.mixture import GaussianMixture as GMM
from IPython import embed as shell
import socket

sys.path.append('/home/vanes/cerebellum_prf/scripts')
from beta_mm import beta_mm_thresh

#######################################################
sub = '01'#sys.argv[1]#'02'
ses ='010203'# sys.argv[2]#'02'
postFix = 'cartfit'#'hrf075_nong'#'_newdm_shifted1'
hrf_delay = 0.75
mask_type = 'gray_matter'
cv = True

if socket.gethostname() == 'aeneas':
    in_home  = os.path.join('/home','shared','2018','visual','cerebellum_prf','derivatives','pp','prf','sub-%s'%sub)
    # in_fn = os.path.join(in_home,'sub-%s_prf_results_zscore_ses_%s_sm3.0_tsnrw%s.nii.gz'%(sub,ses,postFix))
else:
    home = os.path.join('/projects','0','pqsh283','cerprf')
    prf_base_dir = os.path.join(home,'prf')
    z_home = os.path.join(home,'zscore','sub-%s'%sub)
    in_home = os.path.join(prf_base_dir,'sub-%s'%sub)
    volume_mask_home = os.path.join(home,'resources','volume_masks')

if cv:
    cvpostfix = '_cv'
else:
    cvpostfix = ''

if fit_type == 'popeye':
    in_fn = os.path.join(in_home,'sub-%s_%s_%s_hrf%.2f_%s%s.nii.gz'%(sub,out_fn,postFix,hrf_delay,mask_type,cvpostfix))
elif fit_type == 'grid':
    in_fn = os.path.join(in_home,'sub-%s_prf_results_zscore_ses_%s_sm3.0_tsnrw%s%s.nii.gz'%(sub,ses,postFix,cvpostfix))

out_fn = 'new_prf_results_zscore_ses_%s'%ses

# threshes
sizethresh = 15
xthresh = 15
ythresh = 15

#######################################################

# load
img = nb.load(in_fn)
params = img.get_data()

if params.shape[-1] == 8:
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

elif params.shape[-1] ==7:
    dims = {
    'x':0,
    'y':1,
    'ecc':2,
    'ang':3,
    'size':4,
    'r2':5,
    'amp':6,
    }

if params.shape[-1] == 9:
    dims = {
    'x':0,
    'y':1,
    'ecc':2,
    'ang':3,
    'size':4,
    'hrf_delay':5,
    'r2':6,
    'amp':7,
    'n':8
    }

spill_fn = os.path.join(volume_mask_home,'spillovermask_new.nii.gz')
spillmask = nb.load(spill_fn).get_data().astype(bool)

epi_mask_fn = os.path.join(z_home,'mean_over_runs_timemean_ses_03_fnirted.nii.gz')
epi_mask = (nb.load(epi_mask_fn).get_data()>500)
gray_mask_fn = os.path.join(volume_mask_home,'avg152T1_gray.hdr')
gray_mask = np.squeeze((nb.load(gray_mask_fn).get_data()>0.3)) # conservative threshold

mask = gray_mask*epi_mask
# r2thresh=beta_mm_thresh(data=np.ravel(params[mask,dims['r2']]))

r2thresh = 0.12
print('beta mixture model yielded r2 of %.2f'%r2thresh)

# save additional, masked niftis
for apply_cmask in [0]:
    if apply_cmask == 1:
        postFix = '_cmask'
    else:
        postFix = ''

    for amp in ['pos','neg']:
        for val in ['ang']:#,'ecc','size']:

            # add thresholded angles
            these_data = copy.copy(params[:,:,:,dims[val]])#[:,:,:,np.newaxis]

            if apply_cmask == 1:
                these_data[cmask==0] = np.nan
            
            # add spillmask
            these_data[spillmask] = np.nan

            if amp == 'pos':
                these_data[params[:,:,:,dims['amp']]<0] = np.nan
            elif amp == 'neg':
                these_data[params[:,:,:,dims['amp']]>0] = np.nan

            these_data[params[:,:,:,dims['r2']]<r2thresh] = np.nan
            these_data[params[:,:,:,dims['size']]>sizethresh] = np.nan
            these_data[np.abs(params[:,:,:,dims['x']])>xthresh] = np.nan
            these_data[np.abs(params[:,:,:,dims['y']])>ythresh] = np.nan
            prf_nii = nb.Nifti1Image(these_data, affine=img.affine, header=img.header)
            # print 'saving %s'%in_fn.replace('.nii.gz','_ang_%.2f.nii'%r2thresh)
            prf_nii.to_filename(in_fn.replace('.nii.gz','_bmmthresh_%s_%s_%s.nii'%(val,postFix,amp)))

