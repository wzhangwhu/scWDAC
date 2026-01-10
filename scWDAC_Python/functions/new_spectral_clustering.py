import numpy as np
from functions.litekmeans import litekmeans


def new_spectral_clustering(W, numClusters):

    D = np.diag(1.0 / np.sqrt(np.sum(W, axis=1) + np.finfo(float).eps))
    W = D @ W @ D
    U, s, Vt = np.linalg.svd(W, full_matrices=False)
    V = U[:, 0:numClusters]


    _row_norms = np.linalg.norm(V, axis=1, keepdims=True)
    _row_norms = np.maximum(_row_norms, np.finfo(float).eps)
    V = V / _row_norms

    km_out = litekmeans(V, numClusters, 'MaxIter', 100, 'Replicates', 200)
    ids = km_out[0] if isinstance(km_out, (tuple, list)) else km_out

    return ids
