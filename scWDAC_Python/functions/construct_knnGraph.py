from functions.EuDist2 import EuDist2


def construct_knnGraph(X, k, t=None):

    import numpy as np

    n, dim = X.shape
    Dis = EuDist2(X, X)
    t = (np.sum(Dis) / n / n) if (t is None) else t
    idx = np.argsort(Dis, axis=0)
    A_1 = np.zeros((n, n))
    A_1 = A_1 + np.diag(np.ones((n,)))
    for i in range(n):
        id = idx[1:k + 1, i]
        di = Dis[id, i]
        A_1[id, i] = np.exp(-di / (2 * t ** 2))

    H = A_1
    H = H - np.eye(n)
    H = H.T

    return H
