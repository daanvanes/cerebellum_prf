from __future__ import division
import warnings
warnings.simplefilter("ignore")

import numpy as np
from scipy.signal import fftconvolve
from scipy.stats import linregress
import nibabel
from scipy.signal import savgol_filter


from popeye.onetime import auto_attr
import popeye.utilities as utils
from popeye.base import PopulationModel, PopulationFit
from popeye.spinach import generate_og_receptive_field, generate_rf_timeseries_nomask

class CompressiveSpatialSummationModelFiltered(PopulationModel):
    
    r"""
    A Compressive Spatial Summation population receptive field model class
    
    """
    
    def __init__(self, stimulus, hrf_model, cached_model_path=None, nuisance=None, sg_filter_window_length=120, sg_filter_order=3,tr=1.5):
        
        r"""
        A Compressive Spatial Summation population receptive field model [1]_.
        
        Paramaters
        ----------
        
        stimulus : `VisualStimulus` class object
            A class instantiation of the `VisualStimulus` class
            containing a representation of the visual stimulus.
        
        hrf_model : callable
            A function that generates an HRF model given an HRF delay.
            For more information, see `popeye.utilties.double_gamma_hrf_hrf`
        
        References
        ----------
        
        .. [1] Kay KN, Winawer J, Mezer A, Wandell BA (2014) Compressive spatial
        summation in human visual cortex. Journal of Neurophysiology 110:481-494.
        
        """
        
        PopulationModel.__init__(self, stimulus, hrf_model, cached_model_path, nuisance)
        
        self.window = np.int(sg_filter_window_length / tr)

        # Window must be odd
        if self.window % 2 == 0:
            self.window += 1

        self.sg_filter_order = sg_filter_order

    # main method for deriving model time-series
    def generate_ballpark_prediction(self, x, y, sigma, n):#, beta, baseline):
        
        # generate the RF
        rf = generate_og_receptive_field(
            x, y, sigma, self.stimulus.deg_x0, self.stimulus.deg_y0)

        # normalize by the integral
        rf /= ((2 * np.pi * sigma**2) * 1 /
               np.diff(self.stimulus.deg_x0[0, 0:2])**2)

        # extract the stimulus time-series
        response = generate_rf_timeseries_nomask(self.stimulus.stim_arr0, rf)
        
        # compression
        response **= n
        
        # convolve with the HRF
        hrf = self.hrf_model(self.hrf_delay, self.stimulus.tr_length)
        
        # convolve it with the stimulus
        model = fftconvolve(response, hrf)[0:len(response)]

        # units
        model /= np.max(model)

        # at this point, add filtering with a savitzky-golay filter
        model_drift = savgol_filter(model, window_length=self.window, polyorder=self.sg_filter_order,
                                      deriv=0, mode='nearest')
        # demain model_drift, so baseline parameter is still interpretable
        model_drift_demeaned = model_drift-np.mean(model_drift)
        # and apply to data
        model -= model_drift_demeaned
       
        # regress out mean and linear
        p = linregress(model, self.data)
        
        # offset
        model += p[1]
        
        # scale
        model *= p[0]
        
        return model
        
    # main method for deriving model time-series
    def generate_prediction(self, x, y, sigma, n, beta, baseline,unscaled=False):
        
        # generate the RF
        rf = generate_og_receptive_field(
            x, y, sigma, self.stimulus.deg_x, self.stimulus.deg_y)
        
        # normalize by the integral
        rf /= ((2 * np.pi * sigma**2) * 1 /
               np.diff(self.stimulus.deg_x[0, 0:2])**2)
        
        # extract the stimulus time-series
        response = generate_rf_timeseries_nomask(self.stimulus.stim_arr, rf)
        
        # compression
        response **= n
        
        # convolve with the HRF
        hrf = self.hrf_model(self.hrf_delay, self.stimulus.tr_length)
        
        # convolve it with the stimulus
        model = fftconvolve(response, hrf)[0:len(response)]
        
        # units
        model /= np.max(model)
        
        # at this point, add filtering with a savitzky-golay filter
        model_drift = savgol_filter(model, window_length=self.window, polyorder=self.sg_filter_order,
                                      deriv=0, mode='nearest')
        # demain model_drift, so baseline parameter is still interpretable
        model_drift_demeaned = model_drift-np.mean(model_drift)
        
        # and apply to data
        model -= model_drift_demeaned    
        
        # offset
        model += baseline
        
        # scale it by beta
        model *= beta

        return model
