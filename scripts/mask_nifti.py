import nibabel as nb
import os
import numpy as np
import copy
import sys 
from sklearn.mixture import GaussianMixture as GMM
from IPython import embed as shell
#######################################################
sub = sys.argv[1]#'02'
ses = sys.argv[2]#'02'
postFix = 'hrf075_nong'#'_newdm_shifted1'
cv = True
in_home  = os.path.join('/home','shared','2018','visual','cerebellum_prf','derivatives','pp','prf','sub-%s'%sub)
# in_fn = os.path.join(in_home,'sub-%s_prf_results_zscore_ses_%s_sm3.0_tsnrw%s.nii.gz'%(sub,ses,postFix))
if cv:
    cvpostfix = 'cv'
else:
    cvpostfix = ''
in_fn = os.path.join(in_home,'sub-%s_new_prf_results_zscore_ses_%s%s%s.nii.gz'%(sub,ses,postFix,cvpostfix))
print in_fn

# threshes
r2threshes = [0.05,0.1,0.2]
sizethresh = 13
xthresh = 12
ythresh = 6

#######################################################

# load
img = nb.load(in_fn)
params = img.get_data()
print params.shape
# save dims:
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
volume_mask_home = os.path.join('/home','shared','2018','visual','cerebellum_prf','resources','volume_masks')

spill_fn = os.path.join(volume_mask_home,'spillover_mask.nii.gz')
cmask_fn = os.path.join(volume_mask_home,'cmask3.nii.gz')
cmask = (nb.load(cmask_fn).get_data()==1)

retmask_fn = os.path.join(volume_mask_home,'cerebellum_retmaps.nii')
retmask = (nb.load(retmask_fn).get_data()>0)


def gmm_threshold(data,n_components=2,maxrange=100):

    # fit gaussian mixture model to define r2 threshold
    gmm = GMM(n_components = n_components)
    gmm = gmm.fit(np.expand_dims(data,1))

    x = np.linspace(0,maxrange,10000)
    p = gmm.predict_proba(np.expand_dims(x,1))
    

    # logprob, responsibilities = gmm.score_samples(np.expand_dims(x,1))

    # pdf = np.exp(logprob)
    # pdf_individual = responsibilities * pdf[:, np.newaxis]

    thresh = x[np.where(p[:,0]>p[:,1])[0]][0]
    if thresh == 0:
        thresh = x[np.where(p[:,1]>p[:,0])[0]][0]

    return thresh

# shell()
# # load cortex mask
# mni_home  = os.path.join('/home','vanes','bin','fsl','data','standard')

# mask_fn = os.path.join(mni_home,'tissuepriors','avg152T1_gray.hdr')
# mask = np.squeeze((nb.load(mask_fn).get_data()>0.1)) # conservative threshold


# # load cortex mask
# mni_home  = os.path.join('/home','vanes','bin','fsl','data','standard')

# mask_fn = os.path.join(mni_home,'tissuepriors','avg152T1_white.hdr')
# whitemask = np.squeeze((nb.load(mask_fn).get_data()>0.8)) # conservative threshold
# print np.max(params[whitemask,dims['r2']])

# gmmthresh = gmm_threshold(data=np.ravel(params[whitemask,dims['r2']]),n_components=2,maxrange=1)

# save additional, masked niftis
for apply_cmask in [0,1]:
    if apply_cmask == 1:
        postFix = '_cmask'
    else:
        postFix = ''

    for amp in ['pos','neg']:
        for val in ['ang']:#,'ecc','size']:
            for r2thresh in r2threshes:

                # add thresholded angles
                these_data = copy.copy(params[:,:,:,dims[val]])#[:,:,:,np.newaxis]

                if apply_cmask == 1:
                    these_data[cmask==0] = np.nan

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
                prf_nii.to_filename(in_fn.replace('.nii.gz','thresh_%s_%.2f%s_%s.nii'%(val,r2thresh,postFix,amp)))

