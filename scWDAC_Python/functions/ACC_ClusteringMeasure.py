def ACC_ClusteringMeasure(Y, predY):
    import numpy as np
    Y = np.asarray(Y)
    if Y.ndim == 2 and Y.shape[1] != 1:
        Y = Y.T
    predY = np.asarray(predY)
    if predY.ndim == 2 and predY.shape[1] != 1:
        predY = predY.T

    n = len(Y.ravel())
    uY = np.unique(Y)
    nclass = len(uY)

    Y0 = np.zeros((n, 1), dtype=float)

    if nclass != np.max(Y):
        for i in range(1, nclass + 1):
            Y0[(Y == uY[i - 1]).ravel()] = i
        Y = Y0

    uY = np.unique(predY)
    nclass = len(uY)

    predY0 = np.zeros((n, 1), dtype=float)

    if nclass != np.max(predY):
        for i in range(1, nclass + 1):
            predY0[(predY == uY[i - 1]).ravel()] = i
        predY = predY0

# if pred_classnum
    res = bestMap(Y, predY)
    ACC = len(np.where((Y == res).ravel())[0]) / len(Y.ravel())
    result = ACC
    return result


def bestMap(L1, L2):
    import numpy as np
    L1 = np.asarray(L1).ravel(order='F')
    L2 = np.asarray(L2).ravel(order='F')

    if L1.shape != L2.shape:
        raise ValueError('size(L1) must == size(L2)')

    L1 = L1 - np.min(L1) + 1
    L2 = L2 - np.min(L2) + 1

    nClass = int(max(np.max(L1), np.max(L2)))
    G = np.zeros((nClass, nClass), dtype=float)
    for i in range(1, nClass + 1):
        for j in range(1, nClass + 1):
            G[i - 1, j - 1] = len(np.where((L1 == i) & (L2 == j))[0])

# ===========    assign with hungarian method    ======
    c, t = hungarian(-G)
    newL2 = np.zeros_like(L2, dtype=float)
    for i in range(1, nClass + 1):
        newL2[(L2 == i)] = c[i - 1]
    return newL2

def MutualInfo(L1, L2):
    import numpy as np
    L1 = np.asarray(L1).ravel(order='F')
    L2 = np.asarray(L2).ravel(order='F')
    if L1.shape != L2.shape:
        raise ValueError('size(L1) must == size(L2)')
    L1 = L1 - np.min(L1) + 1
    L2 = L2 - np.min(L2) + 1

# ===========    make bipartition graph  ============
    nClass = int(max(np.max(L1), np.max(L2)))
    G = np.zeros((nClass, nClass), dtype=float)
    for i in range(1, nClass + 1):
        for j in range(1, nClass + 1):
            G[i - 1, j - 1] = len(np.where((L1 == i) & (L2 == j))[0]) + np.finfo(float).eps
    sumG = np.sum(G)

# ===========    calculate MIhat
    P1 = np.sum(G, axis=1)
    P1 = P1 / sumG

    P2 = np.sum(G, axis=0)
    P2 = P2 / sumG

    H1 = np.sum(-P1 * np.log2(P1))

    H2 = np.sum(-P2 * np.log2(P2))

    P12 = G / sumG

    PPP = P12 / np.tile(P2.reshape(1, -1), (nClass, 1)) / np.tile(P1.reshape(-1, 1), (1, nClass))

    PPP[np.abs(PPP) < 1e-12] = 1

    MI = np.sum(P12.ravel(order='F') * np.log2(PPP.ravel(order='F')))

    MIhat = MI / max(H1, H2)

    MIhat = np.real(MIhat)

    return MIhat

def hungarian(A):
    import numpy as np

    A = np.asarray(A)
    m, n = A.shape

    if m != n:
        raise ValueError('HUNGARIAN: Cost matrix must be square!')


    orig = A.copy()
    # Reduce matrix.
    A = hminired(A)

    # Do an initial assignment.
    A, C, U = hminiass(A)


    while U[n] != 0:
        LR = np.zeros((n,), dtype=int)
        LC = np.zeros((n,), dtype=int)
        CH = np.zeros((n,), dtype=int)
        RH = np.concatenate([np.zeros((n,), dtype=int), np.array([-1], dtype=int)])
        SLC = []
        r = int(U[n])
        LR[r - 1] = -1
        SLR = [r]
        while True:
            if A[r - 1, n] != 0:
                l = int(-A[r - 1, n])
                if (A[r - 1, l - 1] != 0) and (RH[r - 1] == 0):
                    RH[r - 1] = RH[n]
                    RH[n] = r
                    CH[r - 1] = int(-A[r - 1, l - 1])
            else:
                if RH[n] <= 0:
                    A, CH, RH = hmreduce(A, CH, RH, LC, LR, SLC, SLR)
                r = int(RH[n])
                l = int(CH[r - 1])
                CH[r - 1] = int(-A[r - 1, l - 1])
                if A[r - 1, l - 1] == 0:
                    RH[n] = RH[r - 1]
                    RH[r - 1] = 0
            while LC[l - 1] != 0:
                if RH[r - 1] == 0:
                    if RH[n] <= 0:
                        A, CH, RH = hmreduce(A, CH, RH, LC, LR, SLC, SLR)
                    r = int(RH[n])
                l = int(CH[r - 1])
                CH[r - 1] = int(-A[r - 1, l - 1])
                if A[r - 1, l - 1] == 0:
                    RH[n] = RH[r - 1]
                    RH[r - 1] = 0


            if C[l - 1] == 0:
            # Flip all zeros along the path in LR,LC.
                A, C, U = hmflip(A, C, LC, LR, U, l, r)
                break
            else:
                # Label column l with row r.
                LC[l - 1] = r

                # Add l to the set of labelled columns.
                SLC.append(l)

                # Continue with the row assigned to column l.
                r = int(C[l - 1])

                # Label row r with column l.
                LR[r - 1] = l

                # Add r to the set of labelled rows.
                SLR.append(r)

    # Calculate the total cost.
    T = 0
    for l in range(1, n + 1):
        r = int(C[l - 1])
        if r != 0:
            T = T + orig[r - 1, l - 1]
    return C, T

def hminired(A):
    import numpy as np
    A = np.asarray(A)
    m, n = A.shape

    # Subtract column-minimum values from each column.
    colMin = np.min(A, axis=0)
    A = A - colMin

    # Subtract row-minimum values from each row.
    rowMin = np.min(A.T, axis=0).T
    A = A - rowMin.reshape(-1, 1)

    # Get positions of all zeros.
    i, j = np.where(A == 0)
    i = i + 1
    j = j + 1

    # Extend A to give room for row zero list header column.
    A2 = np.zeros((m, n + 1), dtype=A.dtype)
    A2[:, :n] = A
    A2[0, n] = 0
    A = A2
    for k in range(1, n + 1):

        # Get all column in this row.
        cols = j[(i == k)].reshape(-1)

        # Insert pointers in matrix.
        idx_cols = cols.astype(int)
        A[k - 1, np.concatenate(([n + 1], idx_cols)) - 1] = np.concatenate((-idx_cols, np.array([0], dtype=int)))
    return A

def hminiass(A):
    import numpy as np
    A = np.asarray(A)
    n, np1 = A.shape

    # Initalize return vectors.
    C = np.zeros((n,), dtype=int)
    U = np.zeros((n + 1,), dtype=int)

    # Initialize last/next zero "pointers".
    LZ = np.zeros((n,), dtype=int)
    NZ = np.zeros((n,), dtype=int)
    for i in range(1, n + 1):
    # Set j to first unassigned zero in row i.
        lj = n + 1
        j = int(-A[i - 1, lj - 1])

         # Repeat until we have no more zeros (j==0) or we find a zero
         # in an unassigned column (c(j)==0).
        while (j != 0) and (C[j - 1] != 0):
         # Advance lj and j in zero list.
            lj = j
            j = int(-A[i - 1, lj - 1])

            # Stop if we hit end of list.
            if j == 0:
                break
        if j != 0:
        # We found a zero in an unassigned column.

        # Assign row i to column j.
            C[j - 1] = i

            # Remove A(i,j) from unassigned zero list.
            A[i - 1, lj - 1] = A[i - 1, j - 1]

            # Update next/last unassigned zero pointers.
            NZ[i - 1] = int(-A[i - 1, j - 1])
            LZ[i - 1] = lj

            # Indicate A(i,j) is an assigned zero.
            A[i - 1, j - 1] = 0

        else:
            # We found no zero in an unassigned column.
            # Check all zeros in this row.
            lj = n + 1
            j = int(-A[i - 1, lj - 1])

            # Check all zeros in this row for a suitable zero in another row.
            while j != 0:

            # Check the in the row assigned to this column.
                r = int(C[j - 1])

                # Pick up last/next pointers.
                lm = int(LZ[r - 1])
                m_ = int(NZ[r - 1])

                # Check all unchecked zeros in free list of this row.

                while m_ != 0:
                # Stop if we find an unassigned column.



                    if C[m_ - 1] == 0:
                        break

                    # Advance one step in list.


                    lm = m_
                    m_ = int(-A[r - 1, lm - 1])


                if m_ == 0:
                    # We failed on row r. Continue with next zero on row i.


                    lj = j
                    j = int(-A[i - 1, lj - 1])
                else:
                    # We found a zero in an unassigned column.

                    # Replace zero at (r,m) in unassigned list with zero at (r,j)


                    A[r - 1, lm - 1] = -j
                    A[r - 1, j - 1] = A[r - 1, m_ - 1]

                    # Update last/next pointers in row r.


                    NZ[r - 1] = int(-A[r - 1, m_ - 1])
                    LZ[r - 1] = j

                    # Mark A(r,m) as an assigned zero in the matrix . . .

                    A[r - 1, m_ - 1] = 0

                    # ...and in the assignment vector.

                    C[m_ - 1] = r

                    # Remove A(i,j) from unassigned list.

                    A[i - 1, lj - 1] = A[i - 1, j - 1]

                    # Update last/next pointers in row r.


                    NZ[i - 1] = int(-A[i - 1, j - 1])
                    LZ[i - 1] = lj

                    # Mark A(r,m) as an assigned zero in the matrix . . .

                    A[i - 1, j - 1] = 0

                    # ...and in the assignment vector.

                    C[j - 1] = i

                    # Stop search.

                    break

    # Create vector with list of unassigned rows.

    # Mark all rows have assignment.




    r = np.zeros((n,), dtype=int)
    rows = C[C != 0]
    r[rows - 1] = rows
    empty = (np.where(r == 0)[0] + 1).astype(int)

    # Create vector with linked list of unassigned rows.


    U = np.zeros((n + 1,), dtype=int)
    if empty.size == 0:
        U[n] = 0
    else:
        U[n] = int(empty[0])
        for k in range(0, len(empty) - 1):
            U[int(empty[k]) - 1] = int(empty[k + 1])
        U[int(empty[-1]) - 1] = 0

    return A, C, U

def hmflip(A, C, LC, LR, U, l, r):
    import numpy as np


    n = A.shape[0]


    while True:
    # Move assignment in column l to row r.

        C[l - 1] = r

    # Find zero to be removed from zero list..

    # Find zero before this.

        m = np.where(A[r - 1, :] == -l)[0] + 1

    # Link past this zero.


        m = int(m[0])
        A[r - 1, m - 1] = A[r - 1, l - 1]


        A[r - 1, l - 1] = 0

        # If this was the first zero of the path..

        if LR[r - 1] < 0:




            U[n] = U[r - 1]
            U[r - 1] = 0
            return A, C, U
        else:
            # Move back in this row along the path and get column of next zero.

            l = int(LR[r - 1])

            # Insert zero at (r,l) first in zero list.


            A[r - 1, l - 1] = A[r - 1, n]
            A[r - 1, n] = -l

            # Continue back along the column to get row of next zero in path.

            r = int(LC[l - 1])


def hmreduce(A, CH, RH, LC, LR, SLC, SLR):
    import numpy as np


    n = A.shape[0]

    # Find which rows are covered, i.e. unlabelled.

    coveredRows = (LR == 0)

    # Find which columns are covered, i.e. labelled.

    coveredCols = (LC != 0)



    r = (np.where(~coveredRows)[0] + 1).astype(int)
    c = (np.where(~coveredCols)[0] + 1).astype(int)

    # Get minimum of uncovered elements.

    if (r.size == 0) or (c.size == 0):
        m = 0
    else:
        m = np.min(A[np.ix_(r - 1, c - 1)])

    # Subtract minimum from all uncovered elements.

    if (r.size != 0) and (c.size != 0):
        A[np.ix_(r - 1, c - 1)] = A[np.ix_(r - 1, c - 1)] - m

    # Check all uncovered columns..

    for j in c:
        # ...and uncovered rows in path order..

        for i in SLR:
            # If this is a (new) zero..

            if A[i - 1, j - 1] == 0:
                # If the row is not in unexplored list..

                if RH[i - 1] == 0:
                    # ...insert it first in unexplored list.


                    RH[i - 1] = RH[n]
                    RH[n] = i
                    # Mark this zero as "next free" in this row.

                    CH[i - 1] = j

                # Find last unassigned zero on row I.
                row = A[i - 1, :]
                colsInList = (-row[row < 0]).astype(int)

                # No zeros in the list.
                if colsInList.size == 0:
                    l = n + 1
                else:

                    tmp = colsInList[row[colsInList - 1] == 0]
                    l = int(tmp[0])

                # Append this zero to end of list.
                A[i - 1, l - 1] = -j

    # Add minimum to all doubly covered elements.
    r = (np.where(coveredRows)[0] + 1).astype(int)
    c = (np.where(coveredCols)[0] + 1).astype(int)

    # Take care of the zeros we will remove.
    if (r.size == 0) or (c.size == 0):
        i = np.array([], dtype=int)
        j = np.array([], dtype=int)
    else:
        ii, jj = np.where(A[np.ix_(r - 1, c - 1)] <= 0)
        i = r[ii].astype(int)
        j = c[jj].astype(int)


    for k in range(len(i)):
    # Find zero before this in this row.
        lj = np.where(A[i[k] - 1, :] == -j[k])[0] + 1
        lj = int(lj[0])

        # Link past it.
        A[i[k] - 1, lj - 1] = A[i[k] - 1, j[k] - 1]

        # Mark it as assigned.
        A[i[k] - 1, j[k] - 1] = 0
    if (r.size != 0) and (c.size != 0):
        A[np.ix_(r - 1, c - 1)] = A[np.ix_(r - 1, c - 1)] + m

    return A, CH, RH

def Cal_ARI(c1, c2):
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

    # Expected index (for adjustment)

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

