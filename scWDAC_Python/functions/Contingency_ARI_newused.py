def Contingency_ARI_newused(c1, c2):
    import numpy as np

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
    nc = (n * (n ** 2 + 1) - (n + 1) * nis - (n + 1) * njs + 2 * (nis * njs) / n) / (2 * (n - 1))

    # no. agreements
    A = t1 + t2 - t3

    # no. disagreements
    D = -t2 + t3

    # avoid division by zero; if k=1, define Rand = 0
    # adjusted Rand - Hubert & Arabie 1985

    if t1 == nc:
        ARI = 0
    else:
        ARI = (A - nc) / (t1 - nc)

    # Rand 1971        %Probability of agreement
    RI = A / t1

    # Mirkin 1970    %p(disagreement)
    MI = D / t1

    # Hubert 1977    %p(agree)-p(disagree)
    HI = (A - D) / t1

    return ARI


def Contingency(Mem1, Mem2):

    import numpy as np

    Mem1 = np.asarray(Mem1)
    Mem2 = np.asarray(Mem2)
    if (Mem1.ndim != 1) or (Mem2.ndim != 1):
        raise ValueError('Contingency: Requires two vector arguments')

    Cont = np.zeros((int(np.max(Mem1)), int(np.max(Mem2))), dtype=int)

    for i in range(len(Mem1)):
        Cont[int(Mem1[i]) - 1, int(Mem2[i]) - 1] += 1

    return Cont
