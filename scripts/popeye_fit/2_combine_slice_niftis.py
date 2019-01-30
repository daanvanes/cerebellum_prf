import os
import numpy
import nibabel as nb
from IPython import embed as shell
import numpy as np
####### set these
sub = '02'
ses = '01020304'
postFix = 'cartfit'
out_fn = 'new_prf_results_zscore_ses_%s'%ses
hrf_delay = 0.0
mask_type = 'gray_matter'
################

home = os.path.join('/projects','0','pqsh283','cerprf')
prf_base_dir = os.path.join(home,'prf')
sub_out_dir = os.path.join(prf_base_dir,'sub-%s'%sub)

for k in range(10):
	for sl in range(13):
		
		this_fn = os.path.join(sub_out_dir,'sub-%s_%s_%s_hrf%.2f_%s_sl%d_k%d.nii.gz'%(sub,out_fn,postFix,hrf_delay,mask_type,sl,k))
		if os.path.isfile(this_fn):
			this_img = nb.load(this_fn)

			if not 'combined_data' in locals():
				combined_data = np.zeros((this_img.get_shape()))

			these_data = this_img.get_data()

			print 'loading %s'%this_fn
			sls = np.arange(sl*7,(sl+1)*7)
			combined_data[:,:,sls] = these_data[:,:,sls]

	combined_nii = nb.Nifti1Image(combined_data,this_img.affine,this_img.header)
	opfn = os.path.join(sub_out_dir,'sub-%s_%s_%s_hrf%.2f_%s_k%d.nii.gz'%(sub,out_fn,postFix,hrf_delay,mask_type,k))
	combined_nii.to_filename(opfn)
