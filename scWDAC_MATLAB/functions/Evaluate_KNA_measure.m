%%
% MATLAB code for computing k-NN accuracy
clc
clear
close all
currentFolder = pwd;
addpath(genpath(currentFolder))
addpath('E:\scRNA_seq_sc_ATAC_seq')
dataset={'data1_sim','data2_sim','Sai','BRCA','SNARE','Inhouse','scRNA_ATAC_seq','new_10X10kPBMC'};


for ii = 1:length(dataset)
    load(dataset{ii})
    % import the similarity matrix obtained
    S=importdata(strcat('result_representation_',dataset{ii},'.mat'));

    % ----- KNA ----------
    [U,Sv,~]=svds(S,50); M=U*sqrt(Sv);
    knnAcc = evalLocalStruct(M, true_label, 30);
    result_KNA(ii,:)=[knnAcc, lcmc];
    fprintf('k-NN Acc = %.4f, LCMC = %.4f\n',knnAcc, lcmc);
end
save scWDAC_result_KNA_k_30.mat result_KNA


function [knnAcc] = evalLocalStruct(Z, label, k)
%EVALLOCALSTRUCT 计算 k-NN Acc 与 LCMC
%   Z     : n×d 嵌入矩阵（如 scWDAC 的 M）
%   label : n×1 真实标签
%   k     : 邻居数，默认 30
%
%   knnAcc: k-NN Accuracy
%   lcmc  : Local Continuity Meta-Criterion
%
%  Example:
%       [~,S,V]=svds(W,50); M=U*sqrt(S);
%       [ka,lc]=evalLocalStruct(M,label,30);

if nargin<3
    k=30;
end
n = size(Z,1);
assert(numel(label)==n, 'Size mismatch');

%% ---- 1. k-NN Acc  --------------------------
% 1-1  k nearest neighbor
Dz = squareform(pdist(Z,'euclidean'));
[~, idxZ] = sort(Dz,2);          % 每行升序
idxZ = idxZ(:,2:k+1);           % 去掉自身

%
match = zeros(n,1);
for i = 1:n
    match(i) = mean(label(idxZ(i,:)) == label(i));
end
knnAcc = mean(match);
end



