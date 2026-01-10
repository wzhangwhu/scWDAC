import numpy as np

def retain(M):

    N = M.shape[1]
    MM = np.zeros((N, N), dtype=M.dtype)
    ro = 0.032 + 1 / (0.018 * N - 1.42)


    Ind = np.argsort(-np.abs(M), axis=0)
    S = np.take_along_axis(np.abs(M), Ind, axis=0)

    for i in range(N):
        cL1 = np.sum(S[:, i])
        stop = False
        cSum = 0
        t = 0
        while (not stop):
            t = t + 1
            cSum = cSum + S[t - 1, i]
            if (cSum >= ro * cL1):
                stop = True
                MM[Ind[0:t, i], i] = M[Ind[0:t, i], i]
    return MM
