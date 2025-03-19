"""                                                                                                                                                            
Modules to generate tomographic photometric redshift disributions
"""

import numpy as np
import pylab as pl
import pyccl as ccl
import math
from scipy.integrate import quad
import optparse

### Functions to construct ficudial tomographic n(z) for stage-IV-like WL survey

def smail_redshift(z, zmed, alpha = 2, beta = 1.5):
    """
    Return the Smail et al redshift distribution.

    Arguments:
        z (numpy.array): Redshift (or array or redshifts) to return the distribution for
        zmed (float): Median redshift of the survey  
        alpha (float): Exponent of the z/z_0 factor 
        beta (float): Exponent of the argument of the exponential [exp(-(z/z_0)**beta)]
    Note: Requires alpha > 0 and beta(beta-1) > 0 (alpha > 0 and beta > 1) for normalisation to be correct

    Returns:
        (numpy.array): Value of Smail et al distribution    
    """
    x = 1.412*z/zmed
    norm = math.gamma(1+alpha/(beta*(beta-1)))/beta
    norm *= 1.412/zmed/1.7832005773904782
    
    return norm*(x**alpha)*np.exp(-x**beta)

def nz_true(z_min=0,z_max=3,zmed=0.9):
    """
    Divide Smail et al n(z) into num_z equipopulated bins with vertical boundaries
        i.e. assuming perfect photo-z estimates
    NB. only used to determine support of n(z) of true redshift bins in conv_nz(), kept for legacy purposes

    Arguments:
        z_min (float): minimum redshift of survey
        z_max (float): maximum redshift of survey
        z_med (float): median redshift of survey
    Returns:
        dndz_true (list): list of num_z binned redshift distributions
    """
    z = np.linspace(z_min,z_max,1000)
    nz = smail_redshift(z, zmed) #Unbinned n(z)
    cumz = np.cumsum(np.diff(z)*nz[1:])
    area = cumz[-1]/num_z
    lower = 0
    dndz_true = []
    for I in range(num_z):
        upper = np.interp((I+1)*area, cumz, z[1:])
    
        _nz = nz.copy()
        # Mask out areas outside the bin
        mask = np.logical_or(z<lower,z>upper)
        _nz[mask] = 0.
        # Renormalise
        _nz /= area
    
        dndz_true.append((z,_nz))
    
        lower = upper

    return dndz_true

def p_zphot(z_p,z,sqrt_2pi,cb=1.,zb=0.,sigb=0.05,co=1.,zo=0.1,sigo=0.05,fout=0.1):
    """
    Define the conditional probability for some photometric z_p given some true z, p(z_p|z).
    Functional form and input parameters from Euclid prep VII, equation (115) and table 5, respectively.

    Arguments:
        z_p (float): photometric redshift
        z (float): true (i.e. spectroscopic) redshift
    Returns:
        (float): conditional probability of z_p given z
    """
    # See equation (115) of Euclid prep VII
    coeff=1/(sqrt_2pi*(1+z))
    frac1 = (1-fout)/sigb
    arg1 = -0.5*((z-(cb*z_p)-zb)/(sigb*(1+z)))**2
    frac2 = fout/sigo
    arg2 = -0.5*((z-(co*z_p)-zo)/(sigo*(1+z)))**2
    return coeff*(frac1*np.exp(arg1)+frac2*np.exp(arg2))

def conv_integrand(z_p,z_true,sqrt_2pi):
    """
    Integrand entering convolution of Smail et al n(z) with p(z_p|z).
    See equation (112) of Euclid prep VII.    

    Arguments:
        z_p (float): photometric redshift
        z_true (float): true redshift
    Returns:
        (numpy.array): integrand entering convolution of Smail et al n(z) with p(z_p|z).
    """
    return num_z*smail_redshift(z_true,zmed=0.9)*p_zphot(z_p,z_true,sqrt_2pi)

def conv_nz_integral(z_true,z_min,z_max,sqrt_2pi):
    """
    Convolve Smail et al n(z) with p(z_p|z) -- Euclid prep VII equation (112).
    
    Arguments:
        z_true (float): true redshift
        z_min (float): lower bound of true redshift n(z)
        z_max (float): upper bound of true redshift n(z)
    Returns:
        (float): value of p(z_p|z)-convolved n(z) evaluated at z
    """
    conv_dndz_i_z = quad(conv_integrand,z_min,z_max,args=(z_true,sqrt_2pi))[0]
    return conv_dndz_i_z

def conv_nz():
    """
    Evaluate normalised p(z_p|z)-convolved n(z) as a function of z

    Returns:
        (list): list of num_z p(z_p|z)-convolved redshift distributions
    """
    sqrt_2pi = np.sqrt(2*np.pi)
    dndz = nz_true() # num_z binned n(z) with vertical boundaries
    z_true_arr = dndz[0][0] # support of full n(z)
    len_z = len(z_true_arr)
    z_min = np.min(z_true_arr)
    z_max = np.max(z_true_arr)
    n_i_conv = []
    for I in range(num_z):
        nz_i_true = dndz[I][1] # binned n(z), for bin I
        z_minus = z_true_arr[np.min(np.nonzero(nz_i_true))]
        z_plus = z_true_arr[np.max(np.nonzero(nz_i_true))]
        conv_dndz_i = np.array((z_true_arr,np.zeros(len_z)))
        # Convolve n_i(z) with p(z_p|z) (numerator of Euclid prep VII eq. (112))
        for z_idx in range(len_z):
            conv_dndz_i[1][z_idx] += conv_nz_integral(z_true_arr[z_idx],z_minus,z_plus,sqrt_2pi)
        # Divide by denominator of Euclid prep VII eq. (112)
        conv_dndz_i[1] /= quad(conv_nz_integral,z_min,z_max,args=(z_minus,z_plus,sqrt_2pi))[0]
        n_i_conv.append(conv_dndz_i)
    return n_i_conv


