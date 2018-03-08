import numpy as np
import scipy.sparse as sp
import adolc as ad

def calc_jacobian(*args, **kwargs):
    """Wrapper to adolc utilities for calculation of jacobian.

    Input:
    args: independent variable used to record the tape
    tag: tag of the record to differentiate
    sparse: calculate sparse jacobian. While the jacobian is calculated 
            using sparse utilities the solution is returned as dense matrix.
            It is not efficient but does not matter for small jacobians.
    shape: shape of the jacobian matrix, required only when using sparse.
    """
    try:
        tag = kwargs["tag"]
    except:
        tag = 0

    try:
        sparse = kwargs["sparse"]
    except:
        sparse = True

    if sparse:
        try:
            shape = kwargs["shape"]
        except:
            raise ValueError("'shape' should be passed to calculate sparse jacobian!")

        
        options = np.array([0,0,0,0],dtype=int)
        result = ad.colpack.sparse_jac_no_repeat(tag, *args, options=options)
        nnz = result[0]
        ridx = result[1]
        cidx = result[2]
        values = result[3]
        assert nnz > 0
        jac = sp.csr_matrix((values, (ridx, cidx)), shape=shape)
        jac = jac.toarray()
    else:
        jac = ad.jacobian(tag, *args)
    return jac
