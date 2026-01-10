def NormalizeFea(fea, row=None, norm=None):

    import numpy as np
    import scipy.sparse as sp

    if row is None:
        row = 1

    if norm is None:
        norm = 2
    if norm < 1:
        raise ValueError('It is not a norm when p small than 1!')

    if row:
        nSmp = fea.shape[0]

        if sp.issparse(fea):
            feaNorm = np.asarray(np.abs(fea).power(norm).sum(axis=1)).ravel()
        else:
            feaNorm = np.sum(np.abs(fea) ** norm, axis=1)
        feaNorm = np.maximum(1e-14, feaNorm)


        D = sp.diags(feaNorm ** (-(1.0 / norm)), 0, shape=(nSmp, nSmp), format='csr')
        fea = D @ fea
    else:
        nSmp = fea.shape[1]

        if sp.issparse(fea):
            feaNorm = np.asarray(np.abs(fea).power(norm).sum(axis=0)).ravel()
        else:
            feaNorm = np.sum(np.abs(fea) ** norm, axis=0)
        feaNorm = np.maximum(1e-14, feaNorm)


        D = sp.diags(feaNorm ** (-(1.0 / norm)), 0, shape=(nSmp, nSmp), format='csr')
        fea = fea @ D

    return fea


    if False:
        if row:

            nSmp, mFea = fea.shape

            if sp.issparse(fea):

                fea2 = fea.T

                feaNorm = mynorm(fea2, 1)

                for i in range(nSmp):

                    fea2[:, i] = fea2[:, i] / max(1e-10, feaNorm[i])

                fea = fea2.T
            else:

                feaNorm = (np.sum(fea ** 2, axis=1)) ** 0.5

                fea = fea / feaNorm[:, np.ones((mFea,), dtype=int)]
        else:

            mFea, nSmp = fea.shape

            if sp.issparse(fea):

                feaNorm = mynorm(fea, 1)

                for i in range(nSmp):

                    fea[:, i] = fea[:, i] / max(1e-10, feaNorm[i])
            else:

                feaNorm = (np.sum(fea ** 2, axis=0)) ** 0.5

                fea = fea / feaNorm[np.ones((mFea,), dtype=int), :]
