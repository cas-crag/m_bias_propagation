#!/usr/bin/env python3
"""
Code to estimate bias in C_ls given an m-bias map defined by an RMS of spatial variations (parametrised by rank command line argument)
and characteristic angular scale of variations (parametrised by mbias_ell command line argument)

Uses functions from fisher.py and redshift_dist.py modules
"""
import numpy as np
import pyccl as ccl
import pymaster as nmt
import healpy as hp
import os

from fisher import construct_data_vector_and_covariance, construct_data_vector_labels
from redshift_dist import conv_nz

### INITIALISATION

nside = 4096

npix = hp.nside2npix(nside) # number of pixels in HEALPix map

num_z = 10 # Number of tomoghraphic redshift bins

# Survey parameterisation

fsky = 0.36 #~15,000 deg^2
# Define intrinsic ellipticity variance
sig_e = 0.3**2

# Define the noise. As we have set an equi-populated bin, we simply need to define one noise term, but this
# would need adjusted where each bin might be defined in such a way that there are variations in the number of
# sources across bins.
ndens = 30 # gal/arcmin^2
ndens *= (60*180/np.pi)**2 # per steradian
noise = sig_e*num_z/ndens

# Define the fiducial cosmology (not including evolving dark energy in this case)
fiducial_cosmology = dict(Omega_c=0.2607,
                         Omega_b=0.0490,
                         h=0.6766,
                         sigma8=0.810,
                         w0=-1.,
                         #wa=0.,
                         n_s=0.9665)


### PARSE ARGUMENTS

argopts = optparse.OptionParser("usage: %prog [options] arg1")
argopts.add_option("-r", "--rank", dest="rank", nargs=1, type="int")
argopts.add_option("-l", "--mbias_ell", dest="mbias_ell", nargs=1, type="int")

(options, args) = argopts.parse_args()

rank = options.rank # defines m-bias spatial variations RMS value (integer factor of 0.01)

mbias_ell = options.mbias_ell # defines characteristic angular scale of m-bias variations

mbias_rms = (rank+1)*0.01



### FUNCTIONS TO GENERATE C_ls AND CONVOLVE WITH MODE-COUPLING MATRIX


def get_cls(cosmo, is_fiducial):
    """
    Generate tomographics C_ls and covariance matrix given some cosmology
    Arguments:
        cosmo (dict): dict of cosmological parameter values for PyCCL
        is_fiducial (bool): If True, return C_ls and covariance matrix. If False, return only C_ls
    Returns:
        dv (numpy.array): data vector of tomographic shear C_ls
        cov (numpy.array): diagonal Gaussian covariance matrix of tomographic shear C_ls 
    """
    cosmo = ccl.Cosmology(**cosmo,transfer_function='boltzmann_class')                                                                                                                     

    tracers = []
    for I in range(num_z):
        _z, _nz = dndz[I]
        # CCL can introduce an IntegrationError if the nz's contain too big a range of values.                                                                                    
        # Therefore we not not represent nz less than 0.01% of the max for each bin.                                                                                              
        zmax=np.amax(_z[_nz>1E-4*np.amax(_nz)])
        mask = _z<=zmax
        tracers.append(ccl.WeakLensingTracer(cosmo, dndz = (_z[mask], _nz[mask]), has_shear=True))

    # Construct the cl matrix                                                                                                                                                     
    cls_labels = ["g%d"%I for I in range(num_z)]
    cls = np.empty((num_z,num_z,len(ell_centres)))

    for I in range(num_z):
        for J in range(I,num_z):
            #check if we want no low ell cut for the input cl for namaster, else create cl with no l_min:                                                                         
            cls[I,J,:] = ccl.angular_cl(cosmo, tracers[I], tracers[J], ell_centres)

            # Enforce symmetry here as shear is symmetric                                                                                                                         
            if I != J:
                cls[J,I,:] = cls[I,J,:]
            else:
                cls[I,J,:] += noise

    # Then construct into data vector and block diagonal covariance                                                                                                               
    return_cov = True if is_fiducial else False

    #return cl with no ell_min if nomin==true:                                                                                                                                    
    return construct_data_vector_and_covariance(cls, ell, cls_labels, dv_labels, fsky, return_cov=return_cov)


def synfast(c_ell):
    """
    Generate a HEALPix map with some N_side, given a C_l
    Arguments:
        c_ell (numpy.array): underlying angular power spectrum
    Returns:
        syn_map (numpy.array): HEALPix map
    """
    cl_array = c_ell.reshape(1,-1)

    syn_map = nmt.synfast_spherical(nside,cl_array,[0],seed=1234)[0]

    return syn_map

def m_bias_map(bias_rms,bias_ell):
    """
    Generate HEALPix map of m-bias field with Gaussian profile
    Arguments:
        bias_rms (float): RMS of spatial variations of m-bias field
    Returns:
        (numpy.array):  HEALPix map of m-bias field 
    """
    
    bias_cl=np.zeros(3*nside-1)

    bias_cl_std = 64. # width of m-bias field C_l Gaussian profile -- doesn't affect things much

    # Gaussian-profile m-bias C_l we use to generate the m-bias map
    bias_cl = ((bias_cl_std*np.sqrt(2*np.pi))**(-1))*np.exp(-0.5*((ell_centres-bias_ell)/bias_cl_std)**2)

    # Generate m-bias map using synfast
    delta_m_unnormed=synfast(bias_cl)

    return bias_rms*delta_m_unnormed/np.std(delta_m_unnormed)

def delta_cls(bias_rms,bias_ell):
    """
    Generate the mode-coupling matrix, convolve with the C_ls, and calculate the delta C_ls
    Returns:
        (numpy.array): Vector of delta C_ls
    """

    # Define the ell values on which the data vector is constructed
    # get_cl function uses the middle of the ell bins, so generate ell bin boundaries
    # such that the bin centres will just be the ell values from 0 to 3*nside-1 
    ell = np.arange(-0.5,nside*3+0.5)
    ell_centres = 0.5*(ell[1:]+ell[:-1])

    # Get data vector labels (used in get_cls())
    dv_labels = construct_data_vector_labels(0., num_z,
                                         include_clustering=False, include_ggl=False,
                                        symmetric_shear = True)
    
    # Get the data vector
    # is_fiducial set to False because we don't use the covariance matrix in this script
    dv = get_cls(fiducial_cosmology, is_fiducial = False)

    # Get lengths of data vector and C_ls
    len_dv = len(dv)
    len_cl = len(ell_centres)
    num_cls = int(len(dv)/len_cl)
    # Enforce first 2 multipoles equal to zero in shear C_l
    for cl_idx in range(len_dv):
        if cl_idx == 0 or cl_idx == 1:
            dv_fid[cl_idx] = 0.
        elif cl_idx%len_cl == 0 or cl_idx%len_cl == 1:
            dv_fid[cl_idx] = 0.

    # Generate m_bias map
    delta_m = m_bias_map(bias_rms,bias_ell)

    # Get mode-coupling matrix
    zero = np.zeros(npix)
    # Initialise NmtField object
    f = nmt.NmtField(1+delta_m, [zero, zero], spin=2, n_iter=0)
    # Dummy binning scheme for bandpowers
    b = nmt.NmtBin(nside, nlb=1)
    w = nmt.NmtWorkspace()
    w.compute_coupling_matrix(f, f, b)

    del f, w

    # Convolve data vector with mode-coupling matrix
    
    conv_cl = np.zeros((num_cls,len_cl))

    # Convolve each C_l in the data vector
    # NB we do not consider a redshift-dependent m-bias in this work
    for cl_dv_i in range(num_cls):
        fid_cl_shear = dv[cl_dv_i*len_cl:(cl_dv_i+1)*len_cl]
        cl_coupled = w.couple_cell([fid_cl_shear, # EE
                            fid_cl_shear*0, # EB
                            fid_cl_shear*0, # BE
                            fid_cl_shear*0]) # BB
        conv_cl[cl_dv_i] = cl_coupled[0,:]

    # Flatten array of convolved C_ls to subtract from data vector
    conv_dv = conv_cl.flatten()

    return dv - conv_dv

#main entry point                                                                                                                                              
if __name__ == '__main__':
    """
    main entry point
    """

    cl_folder = "./delta_cells_nside_%d/"%(nside)
    if os.path.isdir(cl_folder):
        os.chdir(cl_folder)
    else:
        os.mkdir(cl_folder)
        os.chdir(cl_folder)

    # Generate tomographic n(z) redshift distributions
    dndz = conv_nz()

    # Calculate biased C_ls
    delta_cl = delta_cls(mbias_rms,mbias_ell)

    delta_cl_str = "delta_cl_nside_%d_rms_%d.txt"%(nside,mbias_rms)

    np.savetxt(delta_cl_str,delta_cl)


