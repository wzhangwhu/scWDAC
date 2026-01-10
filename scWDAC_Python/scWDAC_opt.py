from functions.EProjSimplex_new import EProjSimplex_new
from functions.construct_knnGraph import construct_knnGraph
import numpy as np

def scWDAC_opt(X, lambda1, lambda2, maxIter):

    V = len(X)

    for i in range(V):
        Xi = np.asarray(X[i], dtype=float)
        col_norm = np.sqrt(np.sum(Xi ** 2, axis=0, keepdims=True))
        X[i] = Xi / (np.tile(col_norm, (Xi.shape[0], 1)) + np.finfo(float).eps)


    N = X[0].shape[1]


    D = [None] * V
    for v in range(1, V + 1):
        Dv = np.zeros((N, N), dtype=float)
        Xv = np.asarray(X[v - 1], dtype=float)
        for i in range(1, N + 1):
            for j in range(1, N + 1):
                diff = Xv[:, i - 1] - Xv[:, j - 1]
                Dv[i - 1, j - 1] = (np.linalg.norm(diff)) ** 2
        D[v - 1] = Dv


    rho = 2.8
    miu = 0.01
    miu_max = 1e8

    ## Initialization
    # Calculate P by KNN with Gaussian kernel function
    P = [None] * V
    for v in range(1, V + 1):

        k = 10
        sigma = 1

        H = construct_knnGraph(np.asarray(X[v - 1]).T, k ,None)
        Pv = np.ones(np.shape(H), dtype=float)
        G = np.exp(-D[v - 1] / (2 * (sigma ** 2)))

        for i in range(1, len(H) + 1):
            IDX = np.where(H[i - 1, :] != 0)[0]  # 0-based
            Pv[i - 1, IDX] = G[i - 1, IDX]

        P[v - 1] = Pv

    CC = np.zeros((N, N), dtype=float)


    C = [None] * V
    Z = [None] * V
    S = [None] * V
    L = [None] * V
    R = [None] * V
    E = [None] * V
    Q1 = [None] * V
    Q2 = [None] * V
    Q3 = [None] * V
    w = [None] * V

    for v in range(1, V + 1):
        C[v - 1] = np.zeros((N, N), dtype=float)
        Z[v - 1] = np.zeros((N, N), dtype=float)
        S[v - 1] = np.zeros((N, N), dtype=float)
        L[v - 1] = np.zeros((N, N), dtype=float)
        R[v - 1] = np.zeros((N, N), dtype=float)
        E[v - 1] = np.zeros(np.shape(X[v - 1]), dtype=float)
        Q1[v - 1] = np.zeros(np.shape(X[v - 1]), dtype=float)
        Q2[v - 1] = np.zeros((N, N), dtype=float)
        Q3[v - 1] = np.zeros((N, N), dtype=float)
        w[v - 1] = 1 / V

    ## Interation
    error = np.zeros((maxIter,), dtype=float)
    for iter in range(1, maxIter + 1):

        ## Update Zv
        for v in range(1, V + 1):
            Xv = np.asarray(X[v - 1], dtype=float)
            M1 = Xv - E[v - 1] + Q1[v - 1] / miu
            M2 = L[v - 1] @ C[v - 1] @ R[v - 1].T - Q2[v - 1] / miu
            M3 = S[v - 1] - Q3[v - 1] / miu
            tempZ = Xv.T @ Xv + 2 * np.eye(N)
            Z[v - 1] = np.linalg.solve(tempZ, (Xv.T @ M1 + M2 + M3))

        ## Update Sv
        for v in range(1, V + 1):
            Sv = Z[v - 1] + (Q3[v - 1] - lambda1 * P[v - 1] * D[v - 1]) / miu
            Sv = Sv - np.diag(np.diag(Sv))
            for ic in range(1, N + 1):
                idx = np.arange(1, N + 1)
                idx = idx[idx != ic]  # 1-based values
                # S{v}(ic, idx) = EProjSimplex_new(S{v}(ic, idx));
                row_vals = Sv[ic - 1, idx - 1]
                x_proj, _ft = EProjSimplex_new(row_vals)
                Sv[ic - 1, idx - 1] = x_proj
            S[v - 1] = Sv

        ## Updata CC
        Sw = 0
        SwC = np.zeros((N, N), dtype=float)
        for v in range(1, V + 1):
            Sw = Sw + w[v - 1]
            SwC = SwC + w[v - 1] * C[v - 1]


        tempC = SwC / Sw
        beta = 1 / (2 * Sw)

        Uc, Sc_vec, Vct = np.linalg.svd(tempC, full_matrices=False)
        AC = Uc
        SC = Sc_vec
        VC = Vct.T


        SVP = int(np.sum(SC > beta))

        if SVP >= 1:
            SC_adj = SC[:SVP] - beta
        else:
            SVP = 1
            SC_adj = np.array([0.0], dtype=float)


        CC = AC[:, :SVP] @ np.diag(SC_adj) @ VC[:, :SVP].T

        ## Update Cv
        for v in range(1, V + 1):
            N1 = Z[v - 1] + Q2[v - 1] / miu
            numerator = (miu * (L[v - 1].T @ N1 @ R[v - 1]) + 2 * w[v - 1] * CC)
            denominator = (miu + 2 * w[v - 1])
            C[v - 1] = numerator / denominator

        ## Update Lv and Rv
        for v in range(1, V + 1):
            tempL = (Z[v - 1] + Q2[v - 1] / miu) @ R[v - 1] @ C[v - 1].T
            u, _s, vt = np.linalg.svd(tempL, full_matrices=False)
            L[v - 1] = u @ vt

            tempR = (Z[v - 1] + Q2[v - 1] / miu).T @ L[v - 1] @ C[v - 1]
            u, _s, vt = np.linalg.svd(tempR, full_matrices=False)
            R[v - 1] = u @ vt

        ## Updata Ev
        for v in range(1, V + 1):
            Xv = np.asarray(X[v - 1], dtype=float)
            tempE = Xv - Xv @ Z[v - 1] + Q1[v - 1] / miu
            for i in range(1, N + 1):
                nw = np.linalg.norm(tempE[:, i - 1])
                if nw > (lambda2 / miu):
                    xcol = (nw - lambda2 / miu) * tempE[:, i - 1] / nw
                else:
                    xcol = np.zeros((tempE.shape[0],), dtype=float)
                tempE[:, i - 1] = xcol
            E[v - 1] = tempE

        ## Updata wv
        for v in range(1, V + 1):
            fro = np.linalg.norm((C[v - 1] - CC), ord='fro')
            w[v - 1] = 1 / ((2 * (fro ** 2 + (10 ** (-4)))) ** 0.5)

        ## Update Q1, Q2, Q3, and miu
        for v in range(1, V + 1):
            Xv = np.asarray(X[v - 1], dtype=float)
            tempQ1 = Xv - Xv @ Z[v - 1] - E[v - 1]
            tempQ2 = Z[v - 1] - (L[v - 1] @ C[v - 1] @ R[v - 1].T)
            tempQ3 = Z[v - 1] - S[v - 1]
            Q1[v - 1] = Q1[v - 1] + miu * tempQ1
            Q2[v - 1] = Q2[v - 1] + miu * tempQ2
            Q3[v - 1] = Q3[v - 1] + miu * tempQ3

        # miu = min(rho*miu, miu_max);
        miu = min(rho * miu, miu_max)

        ## Stop
        # % Calculate objection value
        temp_ter1 = np.zeros((V,), dtype=float)
        for v in range(1, V + 1):
            tempstop = Z[v - 1] - (L[v - 1] @ C[v - 1] @ R[v - 1].T)
            temp_ter1[v - 1] = np.max(np.max(np.abs(tempstop)))
        stop = np.max(temp_ter1)
        error[iter - 1] = stop

        print('iter' + str(iter) + 'stop' + str(stop))

        if abs(stop) < 1e-2:
            break


    Zn = np.zeros((N, N), dtype=float)
    for v in range(1, V + 1):
        Zn = Zn + Z[v - 1]

    return Zn
