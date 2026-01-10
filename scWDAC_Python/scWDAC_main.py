import os
import numpy as np
from scipy.io import loadmat

from scWDAC_opt import scWDAC_opt
from functions.retain import retain
from functions.postprocessor import postprocessor
from functions.new_spectral_clustering import new_spectral_clustering
from functions.Clustering8Measure import Clustering8Measure
import time



def scWDAC(X, true_label):
    nCluster = len(np.unique(true_label))
    lambda1 = 0.01
    lambda2 = 0.05
    maxIter = 30

    tic = time.perf_counter()
    Zn = scWDAC_opt(X, lambda1, lambda2, maxIter)
    M = retain(Zn)
    W = postprocessor(M)

    result_label = new_spectral_clustering(W, nCluster)
    time_cost = time.perf_counter() - tic
    result = [*Clustering8Measure(true_label, result_label), time_cost]
    return result


def main():
    data_path = os.path.join("datasets", "data1_sim.mat")
    data = loadmat(data_path)

    X_mat = data["X"]
    true_label = data["true_label"].squeeze()

    if X_mat.shape[0] == 1:
        X = [X_mat[0, i] for i in range(X_mat.shape[1])]
    else:
        X = [X_mat[i, 0] for i in range(X_mat.shape[0])]

    # result = [ACC NMI Fscore Precision ARI Purity Recall Entropy Time]
    result = scWDAC(X, true_label)

    print("result =", result)


if __name__ == "__main__":
    main()
