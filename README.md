# cerebellum_prf repo

This repo includes all code for preprocessing and fitting the pRFs to individual subject data. 

Files necessary for pRF fitting (such as design matrix) can be found in the resources folder.

The data have been preprocessed through fmriprep. This includes motion correction and spatial alignment to MNI space.

The ipynbs perform further preprocessing:

1. 0_register_anats is a notebook used for pre-fmriprep stage 

fmriprep was run for each session (bids dir separately) using this command:

singularity run --bind <BIND_DIRS> <PATH_TO_SINGULARITY.IMG> <PATH_TO_BIDS_DIR> <OUTPUT_DIR> participant --participant_label 01 --nthreads 13 --output-space template --template-resampling-grid 2mm --fs-license-file <PATH_TO_FS_LICENCE> --no-freesurfer -w <WORK_PATH>

2. 1_resample_imgs puts all files in the same grid as used in the HCP data set (91x119x91)
3. 2_improve_spatial alignment uses FNIRT to improve across-session alignment
4. 3_spatial_smoothing applies smoothing kernel of 3 mm to all data
5. 4_sgtf performs high-pass filtering (removing slow drifts)
6. 5_nuissance regression can perform nuissance regression based on the confounds provided by frmriprep (not applied in final analyses)
7. 6_tsnr computes tsnr maps 
8. 7_standardize signal can compute zscore or percent signal change over time
9. 8_tsnr_weighted_avgs creates tsnr-weighted averages over runs. It also creates 10 random train and test sets for later cross-validation

The scripts folder contains a quick 'grid-fit' method that provides fair results in little computation time, and a popeye-fit that finds fine-grained results, but takes much more time to compute.

The submissions folder is used to submit jobs to the SURFSARA Cartesius computing servers.

The beta_mm function can compute an r2 threshold based on fitting 2 beta distributions to the data. It then finds the r2 threshold at which the chance is 0.01 that an r2 higher than that threshold is from the noise distribution.

The alternative_models.py sets up alternative models for the pRF (an onset and a on/off model).

The mask_nifti function creates masked niftis for creation of flatmaps.


