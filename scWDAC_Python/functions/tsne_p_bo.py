def tsne_p_bo(P, labels=None, no_dims=None):

    import numpy as np

    P = np.asarray(P, dtype=float)
    P = P - np.diag(np.diag(P))

    if labels is None:
        labels = []

    if no_dims is None or (hasattr(no_dims, '__len__') and len(np.atleast_1d(no_dims)) == 0):
        no_dims = 2

    # First check whether we already have an initial solution
    if np.size(no_dims) > 1:
        initial_solution = True
        ydata = np.asarray(no_dims, dtype=float)
        no_dims = ydata.shape[1]
    else:
        initial_solution = False

    # Initialize some variables
    n = P.shape[0]
    momentum = 0.08
    final_momentum = 0.1
    mom_switch_iter = 250
    stop_lying_iter = 100
    max_iter = 1000
    epsilon = 500
    min_gain = 0.01

    # Make sure P-vals are set properly
    P.flat[::n + 1] = 0
    P = 0.5 * (P + P.T)

    P = np.maximum(P / np.sum(P), np.finfo(float).tiny)

    const = np.sum(P * np.log(P))

    if not initial_solution:
        P = P * 4

    # Initialize the solution
    if not initial_solution:
        ydata = 0.0001 * np.random.randn(n, int(no_dims))

    y_incs = np.zeros_like(ydata)
    gains = np.ones_like(ydata)

    # Run the iterations
    for iter in range(1, max_iter + 1):

        # Compute joint probability that point i and j are neighbors
        sum_ydata = np.sum(ydata ** 2, axis=1, keepdims=True)
        num = 1.0 / (1.0 + (sum_ydata + sum_ydata.T - 2.0 * (ydata @ ydata.T)))
        num.flat[::n + 1] = 0
        Q = np.maximum(num / np.sum(num), np.finfo(float).tiny)
        # Compute the gradients (faster implementation)
        L = (P - Q) * num
        y_grads = 4.0 * ((np.diag(np.sum(L, axis=0)) - L) @ ydata)
        # Update the solution
        gains = (gains + 0.2) * (np.sign(y_grads) != np.sign(y_incs)) + \
                (gains * 0.8) * (np.sign(y_grads) == np.sign(y_incs))
        gains[gains < min_gain] = min_gain
        y_incs = momentum * y_incs - epsilon * (gains * y_grads)
        ydata = ydata + y_incs
        ydata = ydata - np.mean(ydata, axis=0, keepdims=True)
        ydata[ydata < -100] = -100
        ydata[ydata > 100]  = 100

        # Update the momentum if necessary
        if iter == mom_switch_iter:
            momentum = final_momentum

        if iter == stop_lying_iter and not initial_solution:
            P = P / 4

        # Print out progress
        if iter % 10 == 0:
            cost = const - np.sum(P * np.log(Q))
            print(f'Iteration {iter}: error is {cost}')

        # Display scatter plot (maximally first three dimensions)
        if len(labels) != 0:
            import matplotlib.pyplot as plt
            labels_arr = np.asarray(labels)
            plt.clf()

            if no_dims == 1:
                plt.scatter(ydata[:, 0], ydata[:, 0], s=9, c=labels_arr)

            elif no_dims == 2:
                plt.scatter(ydata[:, 0], ydata[:, 1], s=9, c=labels_arr)

            else:
                ax = plt.gca(projection='3d')
                ax.scatter(ydata[:, 0], ydata[:, 1], ydata[:, 2], s=40, c=labels_arr)

            plt.axis('equal')
            plt.axis('tight')

            plt.pause(0.001)

    return ydata
