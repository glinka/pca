import numpy as np
import scipy.sparse.linalg as spla

def pca(data, k, corr=False):
    """Calculates the top 'k' principal components of the data array and their corresponding variances. This is accomplished via the explicit calculation of the covariance or correlation matrix, depending on which version is desired, and a subsequent partial eigendecomposition.

    Args:
        data (array): a shape ("number of data points", "dimension of data") or (n, m) array containing the data to be operated on
        k (int): the number of principal componenets to be found
        corr (bool): determines whether the correlation matrix will be used instead of the more traditional covariance matrix. Useful when dealing with disparate scales in data measurements as the correlation between to random variables, given by :math:`corr(x,y) = \frac{cov(x,y)}{\sigma_x \sigma_y}`, includes a natural rescaling of units

    Returns:
        pcs (array): shape (m, k) array in which each column is a principal component of 'data'. Thus the projection onto the these coordinates would be given by the (k, n) array np.dot(pcs.T, dat.T)
        variances (array): shape(k,) array containing the 'k' variances corresponding to the 'k' principal components in 'pcs'

    >>> from pca_test import test_pca
    >>> test_pca()
    """
    n = data.shape[0]
    m = data.shape[1]
    # center the data/subtract means
    data = data - np.average(data, 0)
    # calc experimental covariance matrix
    C = np.dot(data.T, data)/(n-1)
    if corr:
        # use correlation matrix
        # extract experimental std. devs.
        variances = np.sqrt(np.diag(C))
        variances.shape = (m, 1)
        # rescale to obtain correlation matrix
        C = C/np.dot(variances, variances.T)
    # calculate eigendecomp with arnoldi/lanczos if k < m, else use eigh for full decomposition
    if k < m:
        variances, pcs  = spla.eigs(C, k=k)
    else:
        variances, pcs = np.linalg.eigh(C)
    return pcs, variances
