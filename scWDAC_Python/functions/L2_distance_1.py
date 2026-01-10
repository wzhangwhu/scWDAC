def L2_distance_1(a, b):

    import numpy as np

    a = np.asarray(a)
    b = np.asarray(b)

    if a.shape[0] == 1:
        a = np.vstack((a, np.zeros((1, a.shape[1]))))
        b = np.vstack((b, np.zeros((1, b.shape[1]))))


    aa = np.sum(a * a, axis=0)
    bb = np.sum(b * b, axis=0)
    ab = a.T @ b
    d = (
        np.tile(aa.reshape(-1, 1), (1, bb.shape[0])) +
        np.tile(bb.reshape(1, -1), (aa.shape[0], 1)) -
        2 * ab
    )

    d = np.real(d)
    d = np.maximum(d, 0)

# % force 0 on the diagonal?
# if (df==1)
# d = d.*(1-eye(size(d)));
# end

    return d
