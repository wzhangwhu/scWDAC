def Cal_NMI_newused(true_labels, cluster_labels):

    import numpy as np
    import scipy.sparse as sp

    true_labels    = np.asarray(true_labels, dtype=float)
    cluster_labels = np.asarray(cluster_labels, dtype=float)

    true_labels    = true_labels - np.min(true_labels) + 1
    cluster_labels = cluster_labels - np.min(cluster_labels) + 1

    if true_labels.ndim == 2 and true_labels.shape[1] > true_labels.shape[0]:
        true_labels = true_labels.T

    if cluster_labels.ndim == 2 and cluster_labels.shape[1] > cluster_labels.shape[0]:
        cluster_labels = cluster_labels.T

    n = len(true_labels.ravel())

    rows = np.arange(1, n + 1)
    tl   = true_labels.ravel()
    cl   = cluster_labels.ravel()

    cat = sp.coo_matrix(
        (np.ones(n), (rows - 1, tl.astype(int) - 1))
    ).tocsr()
    cls = sp.coo_matrix(
        (np.ones(n), (rows - 1, cl.astype(int) - 1))
    ).tocsr()

    cls  = cls.T
    cmat = (cls @ cat).toarray()

    # Total number of data for each true label (CAT), n_i
    # Total number of data for each cluster label (CLS), n_j
    n_i = np.sum(cmat, axis=0)
    n_j = np.sum(cmat, axis=1)

    # Calculate n*n_ij / n_i*n_j
    row, col = cmat.shape

    product = (
        np.tile(n_i.reshape(1, -1), (row, 1)) *
        np.tile(n_j.reshape(-1, 1), (1, col))
    )

    index = np.where(product > 0)

    n = np.sum(cmat)

    product[index] = (n * cmat[index]) / product[index]

    # Sum up n_ij*log()
    index = np.where(product > 0)

    product[index] = np.log(product[index])
    product        = cmat * product

    score = np.sum(product)

    # Divide by sqrt( sum(n_i*log(n_i/n)) * sum(n_j*log(n_j/n)) )
    index = np.where(n_i > 0)[0]
    n_i[index] = n_i[index] * np.log(n_i[index] / n)

    index = np.where(n_j > 0)[0]
    n_j[index] = n_j[index] * np.log(n_j[index] / n)

    denominator = np.sqrt(np.sum(n_i) * np.sum(n_j))

    # Check if the denominator is zero
    if denominator == 0:
        score = 0
    else:
        score = score / denominator

    return score
