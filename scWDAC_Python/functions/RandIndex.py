from functions.Contingency import Contingency


def RandIndex(c1, c2):

    import numpy as np
    import math

    c1 = np.asarray(c1)
    c2 = np.asarray(c2)
    if (c1.ndim != 1) or (c2.ndim != 1):
        raise ValueError('RandIndex: Requires two vector arguments')

    # form contingency matrix
    C = Contingency(c1, c2)

    n = np.sum(C)

    # sum of squares of sums of rows
    nis = np.sum((np.sum(C, axis=1)) ** 2)

    # sum of squares of sums of columns
    njs = np.sum((np.sum(C, axis=0)) ** 2)

    # total number of pairs of entities
    t1 = (n * (n - 1)) / 2

    # sum over rows & columnns of nij^2
    t2 = np.sum(C ** 2)

    t3 = 0.5 * (nis + njs)

    n   = float(n)
    nis = float(nis)
    njs = float(njs)

    nc = (n * (n ** 2 + 1) - (n + 1) * nis - (n + 1) * njs + 2 * (nis * njs) / n) / (2 * (n - 1))

    # no. agreements
    A = t1 + t2 - t3

    # no. disagreements
    D = -t2 + t3

    # avoid division by zero; if k=1, define Rand = 0
    # adjusted Rand - Hubert & Arabie 1985

    if t1 == nc:
        AR = 0
    else:
        AR = (A - nc) / (t1 - nc)

    # Rand 1971        %Probability of agreement
    RI = A / t1

    # Mirkin 1970    %p(disagreement)
    MI = D / t1

    # Hubert 1977    %p(agree)-p(disagree)
    HI = (A - D) / t1

    return AR, RI, MI, HI
