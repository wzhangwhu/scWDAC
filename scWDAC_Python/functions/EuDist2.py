def EuDist2(fea_a, fea_b=None, bSqrt=1):

    import numpy as np
    import scipy.sparse as sp

    if fea_b is None or (hasattr(fea_b, '__len__') and len(fea_b) == 0):
        aa = np.sum(fea_a * fea_a, axis=1)
        ab = fea_a @ fea_a.T
        if sp.issparse(aa):
            aa = aa.toarray()
        D = aa[:, None] + aa[None, :] - 2 * ab
        D[D < 0] = 0
        if bSqrt:
            D = np.sqrt(D)
        D = np.maximum(D, D.T)

    else:
        aa = np.sum(fea_a * fea_a, axis=1)
        bb = np.sum(fea_b * fea_b, axis=1)
        ab = fea_a @ fea_b.T
        if sp.issparse(aa):
            aa = aa.toarray()
            bb = bb.toarray()
        D = aa[:, None] + bb[None, :] - 2 * ab
        D[D < 0] = 0

        if bSqrt:
            D = np.sqrt(D)

    return D
