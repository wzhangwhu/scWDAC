def spw(sigma, Lambda, p):
    import numpy as np

    xi = np.zeros(np.shape(sigma))

    if p == 1:
        _cnt = int(np.sum(np.asarray(sigma) > Lambda))
        for i in range(_cnt):
            xi[i] = sigma[i] - Lambda

    else:
        yu = (2 * Lambda * (1 - p)) ** (1 / (2 - p))
        yu = yu + Lambda * p * (yu ** (p - 1))
        idx = np.where(np.asarray(sigma) > yu)[0]
        if idx.size > 0:
            xi[idx] = np.asarray(sigma)[idx]
            for j in range(3):
                xi[idx] = np.asarray(sigma)[idx] - Lambda * p * (xi[idx] ** (p - 1))

    return xi
