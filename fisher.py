"""
Modules to construct a Fisher matrix, biases and interpretation utils.
"""
from collections import OrderedDict
import numpy as np
import scipy.sparse as sp_spa
import scipy.sparse.linalg as sp_spa_la

_modulelabel = "fisher"

def construct_data_vector_labels(n_number, n_shear,
                                 include_shear = True, include_ggl = True, include_clustering = True,
                                 symmetric_shear = True, symmetric_ggl = True, symmetric_clustering=True,
                                 auto_clustering = False):

    dv_labels = []
    if include_clustering:
        for FG in range(n_number):
            for BG in range(n_number):
                label = "n%d,n%d"%(FG,BG)
                if symmetric_clustering and BG<FG: continue
                if auto_clustering and FG != BG: continue
                dv_labels.append(label)

    if include_ggl:
        for FG in range(n_number):
            for BG in range(n_shear):
                label = "n%d,g%d"%(FG,BG)
                if symmetric_ggl and BG<FG: continue
                dv_labels.append(label)

    if include_shear:
        for FG in range(n_shear):
            for BG in range(n_shear):
                label = "g%d,g%d"%(FG,BG)
                if symmetric_shear and BG<FG: continue
                dv_labels.append(label)

    return dv_labels

def construct_data_vector_and_covariance(cls, larr, cls_labels, dv_labels, fsky,
                                         return_dv = True, return_cov = True):
    """
    Convert Cls for all measures (combinations of observables) to a data vector, and construct the covariance
    for that data vector

    :param cls: 3D ndarray, where the first two dimensions loop over observables (e.g. n0, n1, ..., g0, g1 with
        n number density, g shear and 0..Nz the redshift bin index
    :param larr: Array of ell values which give the bin edges for the ell bin
    :param cls_labels: List of labels for all observables (or tracers in CCL parlance) in the order they are covered
        in cls. E.g. if 2 number tracers and 3 shear: cls_labels = ["n0", "n1", "g0", "g1", "g2"]
    :param dv_labels: List of all data vector labels (measures) to be included. Must be constructed as valid cls_labels
        seperated by a comma. E.g. if we include GGL signal for obs n0 anf g3, then dv must contain "n0,g3". See
        construct_data_vector_labels as a helper for this. All observation labels included as part of dv labels must
        be present in cls_labels
    :param fsky: Fraction of the sky to include (covariance is propto 1/fsky)

    :return:

    Notes:
    - This routine does not yet support ell cuts by measure.

    """

    # Mean of each bin
    ell_centre = 0.5*(larr[1:]+larr[:-1])
    # Width of each bin
    d_ell = np.diff(larr)

    cls_labels = np.array(cls_labels)
    dv_labels = np.array(dv_labels)

    def find_cl_label(comp_label):
        index = np.where(cls_labels==comp_label)[0]
        assert len(index) == 1, "Error finding index"
        return index[0]

    cov_prefactor = 1./((2*ell_centre+1.)*d_ell*fsky)

    #print("cov prefactor: ", 1./((2*ell_centre+1.)*d_ell))

    nmeasures = len(dv_labels)
    dv = np.empty((nmeasures,len(ell_centre)))
    cov = np.empty((nmeasures,nmeasures, len(ell_centre)))

    # Loop over element 1 of the data vector
    for idv1,dv1 in enumerate(dv_labels):
        # Extract out the relevant cls_labels for each observable
        obs_labels_1 = dv1.split(",")
        cl_fg_1 = find_cl_label(obs_labels_1[0])
        cl_bg_1 = find_cl_label(obs_labels_1[1])

        if return_dv: dv[idv1,...] = cls[cl_fg_1, cl_bg_1]

        if return_cov:
            # To construct the covariance, we then need to loop over data vector element 2 (all other dv elements)
            for idv2, dv2 in enumerate(dv_labels):
                obs_labels_2 = dv2.split(",")
                cl_fg_2 = find_cl_label(obs_labels_2[0])
                cl_bg_2 = find_cl_label(obs_labels_2[1])


                cov[idv1, idv2,...] = (cls[cl_fg_1,cl_fg_2,:]*cls[cl_bg_1,cl_bg_2,:]+
                                       cls[cl_fg_1,cl_bg_2,:]*cls[cl_bg_1,cl_fg_2,:])*cov_prefactor

    ret = []
    if return_dv: ret.append(dv.flatten())
    if return_cov: ret.append(get_covar_sparse_from_boxdiag(cov))

    # Unpack if only one return
    if len(ret) == 1: ret = ret[0]

    return ret

def get_covar_sparse_from_boxdiag(cov):
    """
    Take in a 3-ndim cov and place into a 2D SciPy sparse cov by treating as block-diagonal in ell

    :param cov: 3-ndim array. Fisr two label measures (elements of data vector), the thrid nell.
    Assumes the same nell for all tracers.
    """
    ntr = len(cov)
    nell = len(cov[0,0])

    flat_cov = cov.flatten()
    row = np.zeros(len(flat_cov))
    col = np.zeros(len(flat_cov))

    for i in range(len(flat_cov)):
        row[i] = i%nell + int((i-(i%(nell*ntr)))/ntr)
        col[i] = i%(nell*ntr)

    cov_sparse_coo = sp_spa.coo_matrix((flat_cov, (row,col)),shape=(ntr*nell,ntr*nell))

    cov_sparse_csr = sp_spa.csr_matrix(cov_sparse_coo)

    return cov_sparse_csr

def get_covar_full_from_boxdiag(cov):
    """

    Take in a 3-ndim cov and place into a 2D cov by treating as block-diagonal in ell

    :param cov: 3-ndim array. First two label measures (elements of data vector), the third nell.
        Assumes the same nell for all tracers

    Note: this could be edited for tracer dependent ell cut, by setting cov size by looping through all tracers,
    then filling in according to nell for each (assuming cuts applied that way
    """
    ntr = len(cov)
    nell = len(cov[0, 0])
    cov_out = np.zeros([ntr * nell, ntr * nell])
    for i in range(ntr):
        for j in range(ntr):
            cov_out[i * nell:(i + 1) * nell, j * nell:(j + 1) * nell] = np.diag(cov[i, j])
    return cov_out

def get_fisher_matrix(dcl,cv) :
    """
    Compute the Fisher matrix.

    :param cv: Covariance matrix - Scipy CSR sparse matrix (TODO: FORM)
    :param dcl: OrderedDict with keys labelling parameters. Note that each element must contain a Data Vector, with
        length equal to either dimension of Cov

    :returns 2D ndarray containing the fisher matrix, where each dimension is ordered by parameter in the same order
        as dcl.keys() (which is why dcl must be an OrderedDict instance)

    """
    _routinelabel = _modulelabel+".get_fisher_matrix:"

    assert isinstance(dcl, OrderedDict), _routinelabel+" dcl entered must be an OrderedDict"

    parnames_full = list(dcl.keys())
    npar=len(parnames_full)

    fisher=np.zeros([npar,npar])
    for ip1 in range(npar) :
        icvdcl1=sp_spa_la.spsolve(cv, dcl[parnames_full[ip1]])
        for ip2 in range(ip1,npar) :
            fisher[ip1,ip2]=np.dot(icvdcl1, dcl[parnames_full[ip2]])
            if ip1!=ip2:
                # Fisher is by definition symmetric, so fill lower triangle from upper triangle values
                fisher[ip2,ip1]=fisher[ip1,ip2]
    return fisher

def get_bias(dcl,delta,cv,ifish) :
    """
    Get the cosmological parameter bias for a shift in the Cls.
    :param dcl: Cl derivatives
    :param dcl: Derivative of Cls with respect to cosmological parameters (dict, with keywords as given by parna\
mes_full,
        which is populated only if a derivative is taken.)
    :param delta: Change in Cls (biased-fiducial) in form of data vector. Must have the same length as any of th\
e 2
        covariance dimensions
    :param cv: Covariance - scipy CSR sparse matrix
    :param ifish: Inverse Fisher matrix (across cosmological parameters)

    :param 1D ndarray of len==npar, giving bias in Fisher parameters (as defined by keys of dcl) due to shift in
        data vector (given by delta)

    # Note: parnames_full is set of all parameters for which there is a derivative

    """
    _routinelabel = _modulelabel+".get_bias:"
    assert isinstance(dcl, OrderedDict), _routinelabel+" dcl entered must be an OrderedDict"

    parnames_full = list(dcl.keys())
    # Note that in original code, it made the distinction between parnames and parnames full. This was because
    # for the bias parameters, parnames would include the base info ("bz") whilst parnames_full would include
    # "node" info (.e.g "bz1")
    parnames = parnames_full

    npar=len(parnames)
    assert np.sum(ifish.shape)/2 == npar, _routinelabel+" Input (inverse) Fisher shape and Cl derivative [dcl] do not" \
                                " agree in number of parameters. dcl npar: %d. Fisher shape: %s"%(npar, ifish.shape)

    icvdcl=sp_spa_la.spsolve(cv,delta)
    pseudo_fisher=np.zeros(npar)
    for ip1 in range(npar) :
        # Compute the pseudo-fisher, which gives the covariance between delta and all parameters contained in the
        # input fisher matrix (as defined by dcl)
        pseudo_fisher[ip1]=np.dot(icvdcl,dcl[parnames_full[ip1]])
    bias=-np.matmul(ifish,pseudo_fisher)
    return bias


