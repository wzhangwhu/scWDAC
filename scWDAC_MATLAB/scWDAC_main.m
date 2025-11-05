%% scWDAC_main
clc
clear
close all
clear memory
%%
addpath('.\functions\')
addpath('.\datasets\')
currentFolder = pwd;
addpath(genpath(currentFolder))

% Sample*Feature
dataset={'SNARE'}
%%
for ii =1:length(dataset)
    ii
    load(dataset{ii})
    V = length(X);
    nCluster = length(unique(true_label));
    lambda1 = 0.01;
    lambda2 = 0.05;
    maxIter = 30;
    tic
    %%
    Zn = scWDAC_opt(X, lambda1, lambda2, maxIter);
    M = retain(Zn);
    W = postprocessor(M);
    result_label = new_spectral_clustering(W, nCluster);
    time = toc;
    %     result = [ACC NMI Fscore Precision ARI Purity Recall Entropy]
    result(ii,:) = [Clustering8Measure(true_label, result_label), time]
end

