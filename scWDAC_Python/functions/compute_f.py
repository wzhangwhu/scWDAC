def compute_f(T, H):
    import numpy as np
    T = np.asarray(T)
    H = np.asarray(H)
    if len(T) != len(H):
        print(T.shape)
        print(H.shape)
    N = len(T)

    numT = 0
    numH = 0
    numI = 0
    for n in range(1, N + 1):
        Tn = (T[n:]) == T[n - 1]
        Hn = (H[n:]) == H[n - 1]
        numT = numT + np.sum(Tn)
        numH = numH + np.sum(Hn)
        numI = numI + np.sum(Tn * Hn)

    p = 1
    r = 1
    f = 1

    if numH > 0:
        p = numI / numH
    if numT > 0:
        r = numI / numT
    if (p + r) == 0:
        f = 0
    else:
        f = 2 * p * r / (p + r)

    return f, p, r
