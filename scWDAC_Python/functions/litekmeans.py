import numpy as np

def randsample(n, k, replace=False, rng=None):

    if rng is None:
        rng = np.random.default_rng()
    k = int(k)
    n = int(n)
    if k < 0 or n < 0:
        raise ValueError("randsample: n and k must be non-negative")
    if (k > n) and (not replace):
        raise ValueError("randsample: k cannot be larger than n when replace=False")
    return rng.choice(n, size=k, replace=replace)


def litekmeans(X, k, *varargin):
    import numpy as np
    import scipy.sparse as sp

    if k is None:
        raise ValueError("litekmeans:TooFewInputs At least two input arguments required.")


    X = np.asarray(X)
    n, p = X.shape

    pnames = ['distance', 'start', 'maxiter', 'replicates', 'onlinephase', 'clustermaxiter']
    dflts  = ['sqeuclidean', 'sample',  [],        [],          'off',          []]

    eid, errmsg, distance, start, maxit, reps, online, clustermaxit = getargs(pnames, dflts, *varargin)
    if eid is not None and eid != '':
        raise ValueError(f"litekmeans:{eid} {errmsg}")

    if isinstance(distance, str):
        distNames = ['sqeuclidean', 'cosine']
        j = [i for i, name in enumerate(distNames) if name.lower() == distance.lower()]
        if len(j) > 1:
            raise ValueError(f"litekmeans:AmbiguousDistance Ambiguous 'Distance' parameter value:  {distance}.")
        elif len(j) == 0:
            raise ValueError(f"litekmeans:UnknownDistance Unknown 'Distance' parameter value:  {distance}.")
        distance = distNames[j[0]]
    else:
        raise ValueError("litekmeans:InvalidDistance The 'Distance' parameter value must be a string.")

    center = []
    if isinstance(start, str):
        startNames = ['sample', 'cluster']

        j = [i for i, name in enumerate(startNames) if name.lower().startswith(start.lower())]
        if len(j) > 1:
            raise ValueError("litekmeans:AmbiguousStart")
        elif len(j) == 0:
            raise ValueError("litekmeans:UnknownStart")
        elif k is None:
            raise ValueError("litekmeans:MissingK You must specify the number of clusters, K.")
        if j[0] == 1:
            if np.floor(.1 * n) < 5 * k:
                j = [0]
        start = startNames[j[0]]
    elif isinstance(start, (np.ndarray, list, tuple)):
        start = np.asarray(start)
        if start.ndim == 2 and start.shape[1] == p:
            center = start
        elif (start.ndim == 1) or (start.ndim == 2 and (start.shape[0] == 1 or start.shape[1] == 1)):
            idx = start.ravel(order='F').astype(int)


            center = X[idx - 1, :]
        else:
            raise ValueError("litekmeans:MisshapedStart The 'Start' matrix must have the same number of columns as X.")
        if k is None or (isinstance(k, (list, tuple)) and len(k) == 0):
            k = center.shape[0]
        elif k != center.shape[0]:
            raise ValueError("litekmeans:MisshapedStart The 'Start' matrix must have K rows.")
        start = 'numeric'
    else:
        raise ValueError("litekmeans:InvalidStart The 'Start' parameter value must be a string or a numeric matrix or array.")


    if maxit == [] or maxit is None:
        maxit = 100


# subsamples is default 10
    if clustermaxit == [] or clustermaxit is None:
        clustermaxit = 10


    if reps == [] or reps is None or (not (center == [] or isinstance(center, list) and len(center) == 0)):

        if reps == [] or reps is None or (not (center == [])):
            reps = 1

    if not (np.isscalar(k) and np.isreal(k) and k > 0 and (round(k) == k)):
        raise ValueError("litekmeans:InvalidK X must be a positive integer value.")
    elif n < k:
        raise ValueError("litekmeans:TooManyClusters X must have more rows than the number of clusters.")

    bestlabel = []
    sumD = np.zeros((k,), dtype=float)
    bCon = False

    for t in range(1, int(reps) + 1):
        if start == 'sample':
            center = X[randsample(n, k), :]
        elif start == 'cluster':
            Xsubset = X[randsample(n, int(np.floor(.1 * n))), :]
            dump, center, _, _, _ = litekmeans(
                Xsubset, k, *varargin, 'start', 'sample', 'replicates', 1, 'MaxIter', clustermaxit
            )
        elif start == 'numeric':
            pass

        last = 0
        label = 1
        it = 0

        if distance == 'sqeuclidean':
            label = np.ones((n,), dtype=int)
            last = np.zeros((n,), dtype=int)

            while np.any(label != last) and it < maxit:
                last = label.copy()


                bb = np.sum(center * center, axis=1).reshape(1, -1)


                ab = X @ center.T


                D = np.tile(bb, (n, 1)) - 2 * ab


                label = np.argmin(D, axis=1) + 1
                val = D[np.arange(n), label - 1]

                ll = np.unique(label)
                if len(ll) < k:
                    missCluster = np.arange(1, k + 1)
                    missCluster = missCluster[~np.isin(missCluster, ll)]
                    missNum = len(missCluster)

                    aa = np.sum(X * X, axis=1)
                    val2 = aa + val
                    idx = np.argsort(val2)[::-1]
                    label[idx[:missNum]] = missCluster


                rows = np.arange(1, n + 1)
                cols = label
                E = sp.coo_matrix((np.ones(n), (rows - 1, cols - 1)), shape=(n, k)).tocsr()


                sumE = np.array(E.sum(axis=0)).ravel()
                W = sp.diags(1.0 / sumE, 0, shape=(k, k))
                center = (E @ W).T @ X
                center = np.asarray(center)

                it = it + 1

            if it < maxit:
                bCon = True

            if bestlabel is None or len(bestlabel) == 0:
                bestlabel = label.copy()
                bestcenter = center.copy()
                if reps > 1:
                    aa = np.sum(X * X, axis=1)
                    if it >= maxit:
                        bb2 = np.sum(center * center, axis=1)
                        ab2 = X @ center.T
                        D2 = aa.reshape(-1, 1) + bb2.reshape(1, -1) - 2 * ab2
                        D2[D2 < 0] = 0
                    else:
                        D2 = aa.reshape(-1, 1) + D
                        D2[D2 < 0] = 0
                    D2 = np.sqrt(D2)
                    for j in range(1, k + 1):
                        sumD[j - 1] = np.sum(D2[label == j, j - 1])
                    bestsumD = sumD.copy()
                    bestD = D2.copy()
            else:
                aa = np.sum(X * X, axis=1)
                if it >= maxit:
                    bb2 = np.sum(center * center, axis=1)
                    ab2 = X @ center.T
                    D2 = aa.reshape(-1, 1) + bb2.reshape(1, -1) - 2 * ab2
                    D2[D2 < 0] = 0
                else:
                    D2 = aa.reshape(-1, 1) + D
                    D2[D2 < 0] = 0
                D2 = np.sqrt(D2)
                for j in range(1, k + 1):
                    sumD[j - 1] = np.sum(D2[label == j, j - 1])
                if np.sum(sumD) < np.sum(bestsumD):
                    bestlabel = label.copy()
                    bestcenter = center.copy()
                    bestsumD = sumD.copy()
                    bestD = D2.copy()

        elif distance == 'cosine':
            label = np.ones((n,), dtype=int)
            last = np.zeros((n,), dtype=int)

            while np.any(label != last) and it < maxit:
                last = label.copy()
                W = X @ center.T
                label = np.argmax(W, axis=1) + 1
                val = W[np.arange(n), label - 1]

                ll = np.unique(label)
                if len(ll) < k:
                    missCluster = np.arange(1, k + 1)
                    missCluster = missCluster[~np.isin(missCluster, ll)]
                    missNum = len(missCluster)
                    idx = np.argsort(val)
                    label[idx[:missNum]] = missCluster

                rows = np.arange(1, n + 1)
                cols = label
                E = sp.coo_matrix((np.ones(n), (rows - 1, cols - 1)), shape=(n, k)).tocsr()

                sumE = np.array(E.sum(axis=0)).ravel()
                Wdiag = sp.diags(1.0 / sumE, 0, shape=(k, k))
                center = (E @ Wdiag).T @ X
                center = np.asarray(center)

                centernorm = np.sqrt(np.sum(center ** 2, axis=1))
                center = center / centernorm.reshape(-1, 1)

                it = it + 1

            if it < maxit:
                bCon = True

            if bestlabel == []:
                bestlabel = label.copy()
                bestcenter = center.copy()
                if reps > 1:
                    if np.any(label != last):
                        W = X @ center.T
                    D2 = 1 - W
                    for j in range(1, k + 1):
                        sumD[j - 1] = np.sum(D2[label == j, j - 1])
                    bestsumD = sumD.copy()
                    bestD = D2.copy()
            else:
                if np.any(label != last):
                    W = X @ center.T
                D2 = 1 - W
                for j in range(1, k + 1):
                    sumD[j - 1] = np.sum(D2[label == j, j - 1])
                if np.sum(sumD) < np.sum(bestsumD):
                    bestlabel = label.copy()
                    bestcenter = center.copy()
                    bestsumD = sumD.copy()
                    bestD = D2.copy()

    label = bestlabel
    center = bestcenter

    if reps > 1:
        sumD = bestsumD
        D = bestD
    else:


        if distance == 'sqeuclidean':
            aa = np.sum(X * X, axis=1)
            if it >= maxit:
                bb2 = np.sum(center * center, axis=1)
                ab2 = X @ center.T
                D = aa.reshape(-1, 1) + bb2.reshape(1, -1) - 2 * ab2
                D[D < 0] = 0
            else:
                D = aa.reshape(-1, 1) + D
                D[D < 0] = 0
            D = np.sqrt(D)
        elif distance == 'cosine':
            if it >= maxit:
                W = X @ center.T
            D = 1 - W

        for j in range(1, k + 1):
            sumD[j - 1] = np.sum(D[label == j, j - 1])

    return label, center, bCon, sumD, D


def getargs(pnames, dflts, *varargin, nargout=None):














# Initialize some variables
    emsg = ''
    eid = ''
    nparams = len(pnames)
    varargout = list(dflts)
    unrecog = []
    nargs = len(varargin)

# Must have name/value pairs

    if (nargs % 2) != 0:
        eid = 'WrongNumberArgs'
        emsg = 'Wrong number of arguments.'
    else:
# Process name/value pairs

        j = 1
        while j <= nargs:

            pname = varargin[j - 1]


            if not isinstance(pname, str):
                eid = 'BadParamName'
                emsg = 'Parameter name must be text.'
                break



            i = [idx + 1 for idx, pn in enumerate(pnames) if pn.lower() == pname.lower()]

            if len(i) == 0:


                if (nargout is not None) and (nargout > nparams + 2):

                    unrecog.extend([varargin[j - 1], varargin[j]])
                else:
                    eid = 'BadParamName'
                    emsg = 'Invalid parameter name:  %s.' % (pname,)
                    break

            elif len(i) > 1:
                eid = 'BadParamName'
                emsg = 'Ambiguous parameter name:  %s.' % (pname,)
                break
            else:

                varargout[i[0] - 1] = varargin[j]

            j = j + 2

    if (nargout is not None) and (nargout == nparams + 3):
        varargout.append(unrecog)
    elif len(unrecog) > 0 and ((nargout is None) or (nargout <= nparams + 2)):

        eid = eid or 'BadParamName'
        emsg = emsg or ('Invalid parameter name(s): %s' % (', '.join([str(u[0]) for u in unrecog]),))

    return (eid, emsg, *varargout)


