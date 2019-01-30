import numpy as np
from IPython import embed as shell
import re
import os
import scipy.io as io
import nibabel as nb
import time
import sys
sys.path.append('/home/vanes/cerebellum_prf/scripts/')
from create_mask import create_mask

#################################
subs = ['01']
ses = {
'01':'010203',
'02':'01020304',
'03':'010203'}
mask_type = 'gray_matter'
postFix = 'cartfit'
n_jobs = 23
hrf_delays = [0.0,0.75]#,0.75]#,0.75]

mask = create_mask(mask_type)

for hrf_delay in hrf_delays:
	for k in np.arange(1,10):# range(10):
		for sub in subs:
				jobscript = open('evaluate_jobscript_base')
				working_string = jobscript.read()
				jobscript.close()

				RE_dict =  {
				'---sub---': sub, 
				'---ses---': str(ses[sub]), 
				'---mask_type---':mask_type,
				'---hrf_delay---':str(hrf_delay),
				'---postFix---':postFix,
				'---n_jobs---': str(n_jobs), 
				'---k---': str(k)}

				for e in RE_dict:
					rS = re.compile(e)
					working_string = re.sub(rS, RE_dict[e], working_string)
				
				of = open('jobscript', 'w')
				of.write(working_string)
				of.close()

				os.system('sbatch jobscript')


