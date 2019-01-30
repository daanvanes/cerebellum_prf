import os
import nibabel as nb
import numpy as np
def create_mask(mask_type='cerebellum'):

    home = os.path.join('/projects','0','pqsh283','cerprf')
    volume_mask_home = os.path.join(home,'resources','volume_masks')

    if mask_type == 'cerebellum':
        # loading cerebellum mask
        mask_fn = os.path.join(volume_mask_home,'cmask.nii')
        mask = nb.load(mask_fn).get_data().astype(bool)
    elif mask_type == 'gray_matter':
        mask_fn = os.path.join(volume_mask_home,'avg152T1_gray.hdr')
        mask = np.squeeze((nb.load(mask_fn).get_data()>0.1)) # conservative threshold
    elif mask_type == 'wang':
        # load wang atlas
        wang_dir = os.path.join(home,'resources','Wang_prob_retmaps')
        mask= np.zeros_like(valid_voxels).astype(bool)
        for hemi in ['lh','rh']:
            resampled_fn = os.path.join(wang_dir,'maxprob_vol_%s_resampled.nii.gz'%hemi)
            mask = (nb.load(resampled_fn).get_data()>0).astype(bool)
    elif mask_type == 'v123wang':
        # load wang atlas
        wang_dir = os.path.join(home,'resources','Wang_prob_retmaps')
        for hemi in ['lh']:
            resampled_fn = os.path.join(wang_dir,'maxprob_vol_%s_resampled.nii.gz'%hemi)
            mask= np.zeros(nb.load(resampled_fn).get_shape()).astype(bool)
            maskdata = nb.load(resampled_fn).get_data()
            mask[(maskdata!=0)*(maskdata<7)] = True    

    return mask