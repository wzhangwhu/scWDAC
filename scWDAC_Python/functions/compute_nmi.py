def compute_nmi(T, H):
    import numpy as np
    T = np.asarray(T).ravel()
    H = np.asarray(H).ravel()
    N = len(T)
    classes = np.unique(T)
    clusters = np.unique(H)
    num_class = len(classes)
    num_clust = len(clusters)
    D = np.zeros((num_class,), dtype=float)

    for j in range(1, num_class + 1):
        index_class = (T == classes[j - 1])
        D[j - 1] = np.sum(index_class)

    mi = 0
    A = np.zeros((num_clust, num_class), dtype=float)
    avgent = 0
    miarr = np.zeros((num_clust, num_class), dtype=float)
    B = np.zeros((num_clust,), dtype=float)
    for i in range(1, num_clust + 1):
        # number of points in cluster 'i'
        index_clust = (H == clusters[i - 1])
        B[i - 1] = np.sum(index_clust)

        for j in range(1, num_class + 1):
            index_class = (T == classes[j - 1])
            A[i - 1, j - 1] = np.sum(index_class * index_clust)

            if A[i - 1, j - 1] != 0:
                miarr[i - 1, j - 1] = (
                    A[i - 1, j - 1] / N
                    * np.log2(N * A[i - 1, j - 1] / (B[i - 1] * D[j - 1]))
                )

                avgent = avgent - (B[i - 1] / N) * (A[i - 1, j - 1] / B[i - 1]) * np.log2(
                    A[i - 1, j - 1] / B[i - 1]
                )
            else:
                miarr[i - 1, j - 1] = 0
            mi = mi + miarr[i - 1, j - 1]

    class_ent = 0

    for i in range(1, num_class + 1):
        class_ent = class_ent + D[i - 1] / N * np.log2(N / D[i - 1])

    clust_ent = 0
    for i in range(1, num_clust + 1):
        clust_ent = clust_ent + B[i - 1] / N * np.log2(N / B[i - 1])

    nmi = 2 * mi / (clust_ent + class_ent)
    return A, nmi, avgent
