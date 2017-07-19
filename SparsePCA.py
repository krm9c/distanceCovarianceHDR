import time
import sys

import numpy as np
from numpy.lib.stride_tricks import as_strided
from math import sqrt
from scipy import linalg
from sklearn.linear_model import Lasso, lars_path
from joblib import Parallel, delayed

################################################################################
# Utilities to spread load on CPUs
def _gen_even_slices(n, n_packs):
    """Generator to create n_packs slices going up to n.
    Examples
    ========
    >>> list(_gen_even_slices(10, 1))
    [slice(0, 10, None)]
    >>> list(_gen_even_slices(10, 10))
    [slice(0, 1, None), slice(1, 2, None), slice(2, 3, None), slice(3, 4, None), slice(4, 5, None), slice(5, 6, None), slice(6, 7, None), slice(7, 8, None), slice(8, 9, None), slice(9, 10, None)]
    >>> list(_gen_even_slices(10, 5))
    [slice(0, 2, None), slice(2, 4, None), slice(4, 6, None), slice(6, 8, None), slice(8, 10, None)]
    >>> list(_gen_even_slices(10, 3))
    [slice(0, 4, None), slice(4, 7, None), slice(7, 10, None)]

    """
    start = 0
    for pack_num in range(n_packs):
        this_n = n//n_packs
        if pack_num < n % n_packs:
            this_n += 1
        if this_n > 0:
            end = start + this_n
            yield slice(start, end, None)
            start = end

def cpu_count():
    """ Return the number of CPUs.
    """
    # XXX: should be in joblib
    try:
        import multiprocessing
    except ImportError:
        return 1
    return multiprocessing.cpu_count()


################################################################################
# sparsePCA
def _update_V(U, Y, alpha, V=None, Gram=None, method='lars'):
    """ Update V in sparse_pca loop.
        Parameters
        ===========
        V: array, optional
            Initial value of the dictionary, for warm restart
    """
    n_features = Y.shape[1]
    n_atoms = U.shape[1]
    coef = np.empty((n_atoms, n_features))
    if method == 'lars':
        if Gram is None:
            Gram = np.dot(U.T, U)
        err_mgt = np.seterr()
        np.seterr(all='ignore')
        XY = np.dot(U.T, Y)
        for k in xrange(n_features):
            # A huge amount of time is spent in this loop. It needs to be
            # tight.
            _, _, coef_path_ = lars_path(U, Y[:, k], Xy=XY[:, k], Gram=Gram,
                                         alpha_min=alpha, method='lasso')
            coef[:, k] = coef_path_[:, -1]
        np.seterr(**err_mgt)
    else:
        clf = Lasso(alpha=alpha, fit_intercept=False)
        for k in range(n_features):
            # A huge amount of time is spent in this loop. It needs to be
            # tight.
            if V is not None:
                clf.coef_ = V[:,k] # Init with previous value of Vk
            clf.fit(U, Y[:,k], max_iter=1000, tol=1e-8)
            coef[:, k] = clf.coef_
    return coef


def _update_V_parallel(U, Y, alpha, V=None, Gram=None, method='lars', n_jobs=1):
    n_samples, n_features = Y.shape
    if Gram is None:
        Gram = np.dot(U.T, U)
    if n_jobs == 1:
        return _update_V(U, Y, alpha, V=V, Gram=Gram, method=method)
    n_atoms = U.shape[1]
    if V is None:
        V = np.empty((n_atoms, n_features))
    slices = list(_gen_even_slices(n_features, n_jobs))
    V_views = Parallel(n_jobs=n_jobs)(
                delayed(_update_V)(U, Y[:, this_slice], V=V[:, this_slice],
                                alpha=alpha, Gram=Gram, method=method)
                for this_slice in slices)
    for this_slice, this_view in zip(slices, V_views):
        V[:, this_slice] = this_view
    return V


def _update_U(U, Y, V, verbose=False, return_r2=False):
    """ Update U in sparse_pca loop. This function modifies in-place U
        and V.
    """
    n_atoms = len(V)
    n_samples = Y.shape[0]
    R = -np.dot(U, V) # Residuals, computed 'in-place' for efficiency
    R += Y
    # Fortran order, as it makes ger faster
    R = np.asfortranarray(R)
    ger, = linalg.get_blas_funcs(('ger',), (U, V))
    for k in xrange(n_atoms):
        # R <- 1.0 * U_k * V_k^T + R
        R = ger(1.0, U[:, k], V[k, :], a=R, overwrite_a=True)
        U[:, k] = np.dot(R, V[k, :].T)
        # Scale Uk
        norm_square_U = np.dot(U[:, k], U[:, k])
        if norm_square_U < 1e-20:
            if verbose == 1:
                sys.stdout.write("+")
                sys.stdout.flush()
            elif verbose:
                print "Adding new random atom"
            U[:, k] = np.random.randn(n_samples)
            # Setting corresponding coefs to 0
            V[k, :] = 0.0
            U[:, k] /= sqrt(np.dot(U[:, k], U[:, k]))
        else:
            U[:, k] /= sqrt(norm_square_U)
            # R <- -1.0 * U_k * V_k^T + R
            R = ger(-1.0, U[:, k], V[k, :], a=R, overwrite_a=True)
    if return_r2:
        R **= 2
        # R is fortran-ordered. For numpy version < 1.6, sum does not
        # follow the quick striding first, and is thus inefficient on
        # fortran ordered data. We take a flat view of the data with no
        # striding
        R = as_strided(R, shape=(R.size, ), strides=(R.dtype.itemsize,))
        R = np.sum(R)
        #R = np.sum(R, axis=0).sum()
        return U, R
    return U


def sparse_pca(Y, n_atoms, alpha, maxit=100, tol=1e-8, method='lars',
        n_jobs=1, U_init=None, V_init=None, callback=None, verbose=False):
    """
    Compute sparse PCA with n_atoms components
    (U^*,V^*) = argmin 0.5 || Y - U V ||_2^2 + alpha * || V ||_1
                 (U,V)
                with || U_k ||_2 = 1 for all  0<= k < n_atoms
    Notes
    ======
    For better efficiency, Y should be fortran-ordered (10 to 20 percent
    difference in execution time on large data).
    """
    t0 = time.time()
    # Avoid integer division problems
    alpha = float(alpha)
    if n_jobs == -1:
        n_jobs = cpu_count()
    n_samples, n_features = Y.shape

    # Init U and V with SVD of Y
    if U_init is not None and V_init is not None:
        U = np.array(U_init, order='F')
        # Don't copy V, it will happen below
        V = V_init
    else:
        U, S, V = linalg.svd(Y, full_matrices=False)
        V = S[:, np.newaxis] * V
    U = U[:, :n_atoms]
    V = V[:n_atoms, :]

    # Fortran-order V, as we are going to access its row vectors
    V = np.array(V, order='F')
    residuals = 0

    def cost_function():
        return 0.5 * residuals + alpha * np.sum(np.abs(V))

    E = []
    current_cost = np.nan
    if verbose == 1:
        print '[sparse_pca]',
    for ii in xrange(maxit):
        dt = (time.time() - t0)
        if verbose == 1:
            sys.stdout.write(".")
            sys.stdout.flush()
        elif verbose:
            print ("Iteration % 3i "
                "(elapsed time: % 3is, % 4.1fmn, current cost % 7.3f)" %
                    (ii, dt, dt/60, current_cost))
        # Update V
        V = _update_V_parallel(U, Y, alpha/n_samples, V=V, method='lars',\
        n_jobs=n_jobs)
        # Update U
        U, residuals = _update_U(U, Y, V, verbose=verbose, return_r2=True)
        current_cost = cost_function()
        E.append(current_cost)
        if ii > 0:
            dE = E[-2] - E[-1]
            assert(dE >= -tol*E[-1] )
            if dE < tol*E[-1]:
                if verbose == 1:
                    # A line return
                    print ""
                elif verbose:
                    print "--- Convergence reached after %d iterations" % ii
                break
        if ii % 5 == 0 and callback is not None:
            callback(locals())

        U_1, S_1, V_1 = linalg.svd(U, full_matrices=False)
        U_2, S_2, V_2 = linalg.svd(V, full_matrices=False)
        Temp = np.dot(np.dot(np.dot(np.diag(S_1), V_1),U_2), np.diag(S_2))
        U_3, S_3, V_3 = linalg.svd(Temp, full_matrices=False)
    return np.dot(U_1, U_3), S_3, np.dot(V_3,V_2), E
