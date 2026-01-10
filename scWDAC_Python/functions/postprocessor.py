import numpy as np

def postprocessor(Zn):

    uu, s, _ = np.linalg.svd(Zn, full_matrices=False)
    r = int(np.sum(s > 1e-6))
    uu = uu[:, 0:r]
    s = np.diag(s[0:r])
    M = uu @ (s ** (1 / 2))
    _row_norms = np.linalg.norm(M, axis=1, keepdims=True)
    _row_norms = np.maximum(_row_norms, np.finfo(float).eps)
    mm = M / _row_norms
    rs = mm @ mm.T
    W = rs ** 2
    return W
