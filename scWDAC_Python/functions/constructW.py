from scWDAC_Python.functions.EuDist2 import EuDist2
import numpy as np
import scipy.sparse as sp
from scWDAC_Python.functions.NormalizeFea import NormalizeFea


def constructW(fea, options=None):

    bSpeed = 1

    if options is None:
        options = {}

    if 'Metric' in options:

        import warnings
        warnings.warn('This function has been changed and the Metric is no longer be supported')

    if 'bNormalized' not in options:
        options['bNormalized'] = 0


    if 'NeighborMode' not in options:
        options['NeighborMode'] = 'KNN'

    if options['NeighborMode'].lower() in ['knn']:
        # For simplicity, we include the data point itself in the kNN
        if 'k' not in options:
            options['k'] = 5
    elif options['NeighborMode'].lower() in ['supervised']:

        if 'bLDA' not in options:
            options['bLDA'] = 0
        if options['bLDA']:
            options['bSelfConnected'] = 1
        if 'k' not in options:
            options['k'] = 0
        if 'gnd' not in options:
            raise ValueError("Label(gnd) should be provided under 'Supervised' NeighborMode!")
        if (fea is not None) and (len(options['gnd']) != fea.shape[0]):
            raise ValueError("gnd doesn't match with fea!")
    else:
        raise ValueError('NeighborMode does not exist!')


    if 'WeightMode' not in options:
        options['WeightMode'] = 'HeatKernel'

    bBinary = 0
    bCosine = 0
    if options['WeightMode'].lower() in ['binary']:
        bBinary = 1
    elif options['WeightMode'].lower() in ['heatkernel']:
        if 't' not in options:
            nSmp = fea.shape[0]
            if nSmp > 3000:
                D = EuDist2(fea[randsample(nSmp, 3000), :])
            else:
                D = EuDist2(fea)
            options['t'] = np.mean(np.mean(D))
    elif options['WeightMode'].lower() in ['cosine']:
        bCosine = 1
    else:
        raise ValueError('WeightMode does not exist!')


    if 'bSelfConnected' not in options:
        options['bSelfConnected'] = 0


    if 'gnd' in options:
        nSmp = len(options['gnd'])
    else:
        nSmp = fea.shape[0]

    maxM = 62500000
    BlockSize = int(np.floor(maxM / (nSmp * 3)))

    if options['NeighborMode'].lower() == 'supervised':
        Label = np.unique(options['gnd'])
        nLabel = len(Label)

        if options['bLDA']:
            G = np.zeros((nSmp, nSmp), dtype=float)
            for idx in range(1, nLabel + 1):
                classIdx = (np.asarray(options['gnd']) == Label[idx - 1])
                G[np.ix_(classIdx, classIdx)] = 1 / np.sum(classIdx)
            W = sp.csr_matrix(G)
            return W

        if options['WeightMode'].lower() in ['binary']:
            if options['k'] > 0:
                G = np.zeros((nSmp * (options['k'] + 1), 3), dtype=float)
                idNow = 0
                for i in range(1, nLabel + 1):
                    classIdx = np.where(np.asarray(options['gnd']) == Label[i - 1])[0] + 1
                    D = EuDist2(fea[classIdx - 1, :], [], 0)
                    # sort each row
                    idx = np.argsort(D, axis=1) + 1
                    idx = idx[:, :options['k'] + 1]

                    nSmpClass = len(classIdx) * (options['k'] + 1)
                    G[idNow:idNow + nSmpClass, 0] = np.tile(classIdx.reshape(-1, 1), (options['k'] + 1, 1)).ravel(order='F')
                    G[idNow:idNow + nSmpClass, 1] = classIdx[idx.ravel(order='F') - 1]
                    G[idNow:idNow + nSmpClass, 2] = 1
                    idNow = idNow + nSmpClass

                W = sp.coo_matrix((G[:, 2], (G[:, 0].astype(int) - 1, G[:, 1].astype(int) - 1)), shape=(nSmp, nSmp)).tocsr()
                W = W.maximum(W.T)
            else:
                G = np.zeros((nSmp, nSmp), dtype=float)
                for i in range(1, nLabel + 1):
                    classIdx = np.where(np.asarray(options['gnd']) == Label[i - 1])[0]
                    G[np.ix_(classIdx, classIdx)] = 1
                W = sp.csr_matrix(G)

            if not options['bSelfConnected']:
                W = W - sp.diags(W.diagonal(), 0)

            W = sp.csr_matrix(W)
            return W

        elif options['WeightMode'].lower() in ['heatkernel']:
            if options['k'] > 0:
                G = np.zeros((nSmp * (options['k'] + 1), 3), dtype=float)
                idNow = 0
                for i in range(1, nLabel + 1):
                    classIdx = np.where(np.asarray(options['gnd']) == Label[i - 1])[0] + 1
                    D = EuDist2(fea[classIdx - 1, :], [], 0)
                    # sort each row
                    idx = np.argsort(D, axis=1) + 1
                    dump = np.take_along_axis(D, idx - 1, axis=1)
                    idx = idx[:, :options['k'] + 1]
                    dump = dump[:, :options['k'] + 1]
                    dump = np.exp(-dump / (2 * (options['t'] ** 2)))

                    nSmpClass = len(classIdx) * (options['k'] + 1)
                    G[idNow:idNow + nSmpClass, 0] = np.tile(classIdx.reshape(-1, 1), (options['k'] + 1, 1)).ravel(order='F')
                    G[idNow:idNow + nSmpClass, 1] = classIdx[idx.ravel(order='F') - 1]
                    G[idNow:idNow + nSmpClass, 2] = dump.ravel(order='F')
                    idNow = idNow + nSmpClass

                W = sp.coo_matrix((G[:, 2], (G[:, 0].astype(int) - 1, G[:, 1].astype(int) - 1)), shape=(nSmp, nSmp)).tocsr()
            else:
                G = np.zeros((nSmp, nSmp), dtype=float)
                for i in range(1, nLabel + 1):
                    classIdx = np.where(np.asarray(options['gnd']) == Label[i - 1])[0]
                    D = EuDist2(fea[classIdx, :], [], 0)
                    D = np.exp(-D / (2 * (options['t'] ** 2)))
                    G[np.ix_(classIdx, classIdx)] = D
                W = sp.csr_matrix(G)

            if not options['bSelfConnected']:
                W = W - sp.diags(W.diagonal(), 0)

            W = W.maximum(W.T)
            return sp.csr_matrix(W)

        elif options['WeightMode'].lower() in ['cosine']:
            if not options['bNormalized']:
                fea = NormalizeFea(fea)

            if options['k'] > 0:
                G = np.zeros((nSmp * (options['k'] + 1), 3), dtype=float)
                idNow = 0
                for i in range(1, nLabel + 1):
                    classIdx = np.where(np.asarray(options['gnd']) == Label[i - 1])[0] + 1
                    D = fea[classIdx - 1, :] @ fea[classIdx - 1, :].T
                    # sort each row
                    idx = np.argsort(-D, axis=1) + 1
                    dump = np.take_along_axis(D, idx - 1, axis=1)
                    idx = idx[:, :options['k'] + 1]
                    dump = dump[:, :options['k'] + 1]

                    nSmpClass = len(classIdx) * (options['k'] + 1)
                    G[idNow:idNow + nSmpClass, 0] = np.tile(classIdx.reshape(-1, 1), (options['k'] + 1, 1)).ravel(order='F')
                    G[idNow:idNow + nSmpClass, 1] = classIdx[idx.ravel(order='F') - 1]
                    G[idNow:idNow + nSmpClass, 2] = dump.ravel(order='F')
                    idNow = idNow + nSmpClass

                W = sp.coo_matrix((G[:, 2], (G[:, 0].astype(int) - 1, G[:, 1].astype(int) - 1)), shape=(nSmp, nSmp)).tocsr()
            else:
                G = np.zeros((nSmp, nSmp), dtype=float)
                for i in range(1, nLabel + 1):
                    classIdx = np.where(np.asarray(options['gnd']) == Label[i - 1])[0]
                    G[np.ix_(classIdx, classIdx)] = fea[classIdx, :] @ fea[classIdx, :].T
                W = sp.csr_matrix(G)

            if not options['bSelfConnected']:
                W = W - sp.diags(W.diagonal(), 0)

            W = W.maximum(W.T)
            return sp.csr_matrix(W)
        else:
            raise ValueError('WeightMode does not exist!')

    if bCosine and (not options['bNormalized']):
        Normfea = NormalizeFea(fea)

    if (options['NeighborMode'].lower() == 'knn') and (options['k'] > 0):
        if not (bCosine and options['bNormalized']):
            G = np.zeros((nSmp * (options['k'] + 1), 3), dtype=float)
            nBlocks = int(np.ceil(nSmp / BlockSize))

            for i in range(1, nBlocks + 1):
                if i == nBlocks:
                    smpIdx = np.arange((i - 1) * BlockSize + 1, nSmp + 1)
                    dist = EuDist2(fea[smpIdx - 1, :], fea, 0)

                    if bSpeed:
                        nSmpNow = len(smpIdx)
                        dump = np.zeros((nSmpNow, options['k'] + 1), dtype=float)
                        idx = np.zeros_like(dump, dtype=int)
                        for j in range(1, options['k'] + 2):
                            idxj = np.argmin(dist, axis=1) + 1
                            dumpj = dist[np.arange(nSmpNow), idxj - 1]
                            dump[:, j - 1] = dumpj
                            idx[:, j - 1] = idxj
                            temp = (idxj - 1) * nSmpNow + (np.arange(1, nSmpNow + 1))
                            dist.ravel(order='F')[temp - 1] = 1e100
                    else:
                        idx = np.argsort(dist, axis=1) + 1
                        dump = np.take_along_axis(dist, idx - 1, axis=1)
                        idx = idx[:, :options['k'] + 1]
                        dump = dump[:, :options['k'] + 1]

                    if not bBinary:
                        if bCosine:
                            dist2 = Normfea[smpIdx - 1, :] @ Normfea.T
                            dist2 = np.asarray(dist2)
                            linidx = np.arange(1, idx.shape[0] + 1).reshape(-1, 1)
                            dump = dist2[linidx - 1, idx - 1]
                        else:
                            dump = np.exp(-dump / (2 * (options['t'] ** 2)))

                    startRow = (i - 1) * BlockSize * (options['k'] + 1)
                    endRow = nSmp * (options['k'] + 1)
                    G[startRow:endRow, 0] = np.tile(smpIdx.reshape(-1, 1), (options['k'] + 1, 1)).ravel(order='F')
                    G[startRow:endRow, 1] = idx.ravel(order='F')
                    if not bBinary:
                        G[startRow:endRow, 2] = dump.ravel(order='F')
                    else:
                        G[startRow:endRow, 2] = 1

                else:
                    smpIdx = np.arange((i - 1) * BlockSize + 1, i * BlockSize + 1)
                    dist = EuDist2(fea[smpIdx - 1, :], fea, 0)

                    if bSpeed:
                        nSmpNow = len(smpIdx)
                        dump = np.zeros((nSmpNow, options['k'] + 1), dtype=float)
                        idx = np.zeros_like(dump, dtype=int)
                        for j in range(1, options['k'] + 2):
                            idxj = np.argmin(dist, axis=1) + 1
                            dumpj = dist[np.arange(nSmpNow), idxj - 1]
                            dump[:, j - 1] = dumpj
                            idx[:, j - 1] = idxj
                            temp = (idxj - 1) * nSmpNow + (np.arange(1, nSmpNow + 1))
                            dist.ravel(order='F')[temp - 1] = 1e100
                    else:
                        idx = np.argsort(dist, axis=1) + 1
                        dump = np.take_along_axis(dist, idx - 1, axis=1)
                        idx = idx[:, :options['k'] + 1]
                        dump = dump[:, :options['k'] + 1]

                    if not bBinary:
                        if bCosine:
                            dist2 = Normfea[smpIdx - 1, :] @ Normfea.T
                            dist2 = np.asarray(dist2)
                            linidx = np.arange(1, idx.shape[0] + 1).reshape(-1, 1)
                            dump = dist2[linidx - 1, idx - 1]
                        else:
                            dump = np.exp(-dump / (2 * (options['t'] ** 2)))

                    startRow = (i - 1) * BlockSize * (options['k'] + 1)
                    endRow = i * BlockSize * (options['k'] + 1)
                    G[startRow:endRow, 0] = np.tile(smpIdx.reshape(-1, 1), (options['k'] + 1, 1)).ravel(order='F')
                    G[startRow:endRow, 1] = idx.ravel(order='F')
                    if not bBinary:
                        G[startRow:endRow, 2] = dump.ravel(order='F')
                    else:
                        G[startRow:endRow, 2] = 1

            W = sp.coo_matrix((G[:, 2], (G[:, 0].astype(int) - 1, G[:, 1].astype(int) - 1)), shape=(nSmp, nSmp)).tocsr()
        else:
            G = np.zeros((nSmp * (options['k'] + 1), 3), dtype=float)
            nBlocks = int(np.ceil(nSmp / BlockSize))

            for i in range(1, nBlocks + 1):
                if i == nBlocks:
                    smpIdx = np.arange((i - 1) * BlockSize + 1, nSmp + 1)
                    dist = fea[smpIdx - 1, :] @ fea.T
                    dist = np.asarray(dist)

                    if bSpeed:
                        nSmpNow = len(smpIdx)
                        dump = np.zeros((nSmpNow, options['k'] + 1), dtype=float)
                        idx = np.zeros_like(dump, dtype=int)
                        for j in range(1, options['k'] + 2):
                            idxj = np.argmax(dist, axis=1) + 1
                            dumpj = dist[np.arange(nSmpNow), idxj - 1]
                            dump[:, j - 1] = dumpj
                            idx[:, j - 1] = idxj
                            temp = (idxj - 1) * nSmpNow + (np.arange(1, nSmpNow + 1))
                            dist.ravel(order='F')[temp - 1] = 0
                    else:
                        idx = np.argsort(-dist, axis=1) + 1
                        dump = np.take_along_axis(dist, idx - 1, axis=1)
                        idx = idx[:, :options['k'] + 1]
                        dump = dump[:, :options['k'] + 1]

                    startRow = (i - 1) * BlockSize * (options['k'] + 1)
                    endRow = nSmp * (options['k'] + 1)
                    G[startRow:endRow, 0] = np.tile(smpIdx.reshape(-1, 1), (options['k'] + 1, 1)).ravel(order='F')
                    G[startRow:endRow, 1] = idx.ravel(order='F')
                    G[startRow:endRow, 2] = dump.ravel(order='F')
                else:
                    smpIdx = np.arange((i - 1) * BlockSize + 1, i * BlockSize + 1)
                    dist = fea[smpIdx - 1, :] @ fea.T
                    dist = np.asarray(dist)

                    if bSpeed:
                        nSmpNow = len(smpIdx)
                        dump = np.zeros((nSmpNow, options['k'] + 1), dtype=float)
                        idx = np.zeros_like(dump, dtype=int)
                        for j in range(1, options['k'] + 2):
                            idxj = np.argmax(dist, axis=1) + 1
                            dumpj = dist[np.arange(nSmpNow), idxj - 1]
                            dump[:, j - 1] = dumpj
                            idx[:, j - 1] = idxj
                            temp = (idxj - 1) * nSmpNow + (np.arange(1, nSmpNow + 1))
                            dist.ravel(order='F')[temp - 1] = 0
                    else:
                        idx = np.argsort(-dist, axis=1) + 1
                        dump = np.take_along_axis(dist, idx - 1, axis=1)
                        idx = idx[:, :options['k'] + 1]
                        dump = dump[:, :options['k'] + 1]

                    startRow = (i - 1) * BlockSize * (options['k'] + 1)
                    endRow = i * BlockSize * (options['k'] + 1)
                    G[startRow:endRow, 0] = np.tile(smpIdx.reshape(-1, 1), (options['k'] + 1, 1)).ravel(order='F')
                    G[startRow:endRow, 1] = idx.ravel(order='F')
                    G[startRow:endRow, 2] = dump.ravel(order='F')

            W = sp.coo_matrix((G[:, 2], (G[:, 0].astype(int) - 1, G[:, 1].astype(int) - 1)), shape=(nSmp, nSmp)).tocsr()

        if bBinary:

            W.data[:] = 1

        if ('bSemiSupervised' in options) and options['bSemiSupervised']:
            tmpgnd = np.asarray(options['gnd'])[np.asarray(options['semiSplit']).astype(bool)]

            Label = np.unique(tmpgnd)
            nLabel = len(Label)
            Gsup = np.zeros((np.sum(options['semiSplit']), np.sum(options['semiSplit'])), dtype=float)
            for idx in range(1, nLabel + 1):
                classIdx = (tmpgnd == Label[idx - 1])
                Gsup[np.ix_(classIdx, classIdx)] = 1
            Wsup = sp.csr_matrix(Gsup)

            if 'SameCategoryWeight' not in options:
                options['SameCategoryWeight'] = 1

            semi = np.asarray(options['semiSplit']).astype(bool)

            block = (Wsup > 0).astype(float) * options['SameCategoryWeight']
            W = W.tolil()
            W[np.ix_(semi, semi)] = block
            W = W.tocsr()

        if not options['bSelfConnected']:

            W = W - sp.diags(W.diagonal(), 0)

        if ('bTrueKNN' in options) and options['bTrueKNN']:
            pass
        else:

            W = W.maximum(W.T)

        return W



    if options['WeightMode'].lower() in ['binary']:
        raise ValueError('Binary weight can not be used for complete graph!')
    elif options['WeightMode'].lower() in ['heatkernel']:
        W = EuDist2(fea, [], 0)
        W = np.exp(-W / (2 * (options['t'] ** 2)))
    elif options['WeightMode'].lower() in ['cosine']:
        W = (Normfea @ Normfea.T)
        W = np.asarray(W)
    else:
        raise ValueError('WeightMode does not exist!')

    if not options['bSelfConnected']:
        for i in range(W.shape[0]):
            W[i, i] = 0

    W = np.maximum(W, W.T)
    return sp.csr_matrix(W)


def randsample(n, k):
    import numpy as np

    return np.random.choice(np.arange(1, n + 1), size=int(k), replace=False) - 1
