import nibabel as nb
import os
import numpy as np
import copy
import sys 
from IPython import embed as shell

repo_dir = os.path.join('/home','vanes','cerebellum_prf')# this should point to your repo 
sys.path.append(repo_dir,'scripts') 
from beta_mm import beta_mm_thresh

########################################################################################
# set dirs
########################################################################################

# this should point to your prf base dir
volume_mask_home = os.path.join(repo_dir,'resources','volume_masks')
# this should point to your resampled files dir where the average functional file is located:
res_home = os.path.join('/home','shared','2018','visual','cerebellum_prf','derivatives','pp','res')
# this should point to your prf results dir:
prf_base_dir = os.path.join('/home','shared','2018','visual','cerebellum_prf','derivatives','pp','prf')

########################################################################################
# set parameters
########################################################################################

subs = ['03']#,'02','03']#'03'#sys.argv[1]#'02'
sess = {
    '01':'010203',
    '02':'01020304',
    '03':'010203',
}

postFix = 'cartfit
hrf_delay = 0.75
mask_type = 'gray_matter'
cv = True
fit_type ='popeye'

compute_r2thresh = False
r2threshes = {
    '01':0.12,
    '02':0.16,
    '03':0.29,
}

diff_performance = 0.05 # this is the CVR2 improvement of the pRF vs the bar-on model

if cv:
    cvpostfix = '_cv'
else:
    cvpostfix = ''

########################################################################################
# code
########################################################################################

for sub in subs:

    # specify dirs:
    in_home  = os.path.join(prf_base_dir,'sub-%s'%sub)
    p_res_home  = os.path.join(res_home,'sub-%s'%sub)

    ses = sess[sub]

    out_fn = 'new_prf_results_zscore_ses_%s'%ses

    if fit_type == 'popeye':
        in_fn = os.path.join(in_home,'sub-%s_%s_%s_hrf%.2f_%s%s.nii.gz'%(sub,out_fn,postFix,hrf_delay,mask_type,cvpostfix))
    elif fit_type == 'grid':
        in_fn = os.path.join(in_home,'sub-%s_prf_results_zscore_ses_%s_sm3.0_tsnrw%s%s.nii.gz'%(sub,ses,postFix,cvpostfix))

    # load
    img = nb.load(in_fn)
    params = img.get_data()

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

    # load in subject specific cortical spill mask
    spill_fn = os.path.join(volume_mask_home,'spillovermask_sub%s.nii.gz'%sub)
    spillmask = nb.load(spill_fn).get_data().astype(bool)

    # create epi mask
    epi_mask_fn = os.path.join(p_res_home,'mean_over_runs_timemean_ses_03_fnirted.nii.gz')
    epi_mask = (nb.load(epi_mask_fn).get_data()>500)

    # create gray matter mask
    gray_mask_fn = os.path.join(volume_mask_home,'avg152T1_gray.hdr')
    gray_mask = np.squeeze((nb.load(gray_mask_fn).get_data()>0.3))

    # combine masks
    mask = gray_mask*epi_mask

    # get the model on prediction
    baron_fn = os.path.join(in_home,'baron_r2diff.nii.gz')
    baron_nii = nb.load(baron_fn)
    visually_unselective = (baron_nii.get_data()>-diff_performance)

    # now compute r2thresh if asked:
    if compute_r2thresh:

        data=np.ravel(params[mask,dims['r2']])
        data=data[~np.isnan(data)]
        threshes = []
        for i in range(10):
            these_data = np.random.choice(data,int(1e4))
            r2thresh=beta_mm_thresh(these_data)
            threshes.append(r2thresh)
            print('beta mixture model yielded r2 of %.2f'%r2thresh)
        r2thresh = np.mean(threshes)
        print('average: r2 of %.2f'%r2thresh)
    else:
        r2thresh = r2threshes[sub]

    # save additional, masked niftis    
    for apply_spillmask in [0,1]:
        if apply_spillmask == 1:
            postFix = '_spillmask'
        else:
            postFix = ''

        for amp in ['pos','neg']:
            for val in ['ang','ecc','size']:

                # add thresholded angles
                these_data = copy.copy(params[:,:,:,dims[val]])#[:,:,:,np.newaxis]

                # add spillmask
                if apply_spillmask == 1:
                    these_data[spillmask] = np.nan

                if amp == 'pos':
                    these_data[params[:,:,:,dims['amp']]<0] = np.nan
                elif amp == 'neg':
                    these_data[params[:,:,:,dims['amp']]>0] = np.nan

                these_data[visually_unselective] = np.nan
                these_data[params[:,:,:,dims['r2']]<r2thresh] = np.nan
                
                prf_nii = nb.Nifti1Image(these_data, affine=img.affine, header=img.header)
                prf_nii.to_filename(in_fn.replace('.nii.gz','_thresh_%s_%s_%s.nii'%(val,postFix,amp)))

