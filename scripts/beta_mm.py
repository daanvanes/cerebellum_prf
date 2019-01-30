
from scipy import stats
import numpy as np
import scipy as sp
from IPython import embed as shell

def beta_mix_ll(x, a1, b1, a2, b2, w):
    
    w1 = w
    w2 = 1 - w1
    
    return np.sum(np.log(w1 * stats.beta(a1, b1).pdf(x) + w2*stats.beta(a2, b2).pdf(x)))

def beta_mm_thresh(data,q=0.01):
    # results = sp.optimize.minimize(lambda pars: -beta_mix_ll(data, *pars), [1.5 ,  50 ,   1,   2 ,   0.5], 
    #                  bounds=([0.01, 100], [0.01, 100], [0.01, 100], [0.01, 100], [0.5, 1]))
    # shell()
    results = sp.optimize.differential_evolution(lambda pars: -beta_mix_ll(data, *pars), bounds=([0, 100], [0, 100], [0, 100], [0, 100], [.5,1]))
    shell()

    a1, b1, a2, b2, w = results.x
    w1 = w
    w2 = 1 - w1
    pd_noise = w1 * stats.beta(a1, b1).pdf(data)
    pd_signal = w2 * stats.beta(a2, b2).pdf(data)
    p_noise = pd_noise/(pd_noise+pd_signal)
    thresholds = np.linspace(0,1,int(1e4))
    thresh = thresholds[np.nanargmin(np.abs(np.array([np.mean(p_noise[data>t]) for t in thresholds])-q))]

    return thresh