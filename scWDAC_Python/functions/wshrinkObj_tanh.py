def wshrinkObj_tanh(x, rho, sX, isWeight, mode=None, alfa=None, theta=None):
    import numpy as np

    if isWeight == 1:
        C = np.sqrt(sX[2] * sX[1])

    if mode is None:
        mode = 1

    X = np.asarray(x).reshape(tuple(sX), order='F')

    if mode == 1:
        Y = X2Yi(X, 3)
    elif mode == 3:
        Y = np.transpose(X, tuple(np.roll(np.arange(X.ndim), -1)))
    else:
        Y = X

    Yhat = np.fft.fft(Y, axis=2)

    objV = 0

    if mode == 1:
        n3 = sX[1]
    elif mode == 3:
        n3 = sX[0]
    else:
        n3 = sX[2]

    if isinstance(n3 / 2, (np.integer, int)):
        endValue = np.int16(n3 / 2 + 1)
        for i in range(1, int(endValue) + 1):

            Yi = np.asarray(Yhat[:, :, i - 1])
            uhat, svals, vhat_t = np.linalg.svd(Yi, full_matrices=False)
            shat = np.diag(svals)
            vhat = vhat_t.T

            if isWeight:
                weight = C / (np.diag(shat) + np.finfo(float).eps)
                tau = rho * weight
                # shat = soft(shat,diag(tau));
                shat = soft_threshold(shat, np.diag(tau))
            else:
                tau = rho
                shat = np.maximum(shat - tau, 0)

            objV = objV + np.sum(shat)
            Yhat[:, :, i - 1] = uhat @ shat @ vhat.conj().T
            if i > 1:
                Yhat[:, :, int(n3 - i + 1)] = uhat.conj() @ shat @ vhat.conj().T
                objV = objV + np.sum(shat)

        Yi = np.asarray(Yhat[:, :, int(endValue)])
        uhat, svals, vhat_t = np.linalg.svd(Yi, full_matrices=False)
        shat = np.diag(svals)
        vhat = vhat_t.T

        if isWeight:
            weight = C / (np.diag(shat) + np.finfo(float).eps)
            tau = rho * weight
            # shat = soft(shat,diag(tau));
            shat = soft_threshold(shat, np.diag(tau))
        else:
            tau = rho
            shat = np.maximum(shat - tau, 0)

        objV = objV + np.sum(shat)
        Yhat[:, :, int(endValue)] = uhat @ shat @ vhat.conj().T

    else:
        endValue = np.int16(n3 / 2 + 1)
        for i in range(1, int(endValue) + 1):

            Yi = np.asarray(Yhat[:, :, i - 1])
            uhat, svals, vhat_t = np.linalg.svd(Yi, full_matrices=False)
            shat = np.diag(svals)
            vhat = vhat_t.T

            if isWeight:
                weight = C / (np.diag(shat) + np.finfo(float).eps)
                tau = rho * weight
                # shat = soft(shat,diag(tau));
                shat = soft_threshold(shat, np.diag(tau))
            else:
                sigma = np.diag(shat)
                w = alfa * (1 - theta * np.tanh(theta * sigma) ** 2)
                tau = rho * w
                shat = np.maximum(shat - np.diag(tau), 0)

            objV = objV + np.sum(shat)
            Yhat[:, :, i - 1] = uhat @ shat @ vhat.conj().T
            if i > 1:
                Yhat[:, :, int(n3 - i + 1)] = uhat.conj() @ shat @ vhat.conj().T
                objV = objV + np.sum(shat)

    Y = np.fft.ifft(Yhat, axis=2)

    if mode == 1:
        X = Yi2X(Y, 3)
    elif mode == 3:
        X = np.transpose(Y, tuple(np.roll(np.arange(Y.ndim), -2)))
    else:
        X = Y

    x = np.asarray(X).ravel(order='F')

    return x, objV
