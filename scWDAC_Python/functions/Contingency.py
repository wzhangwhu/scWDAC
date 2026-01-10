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
