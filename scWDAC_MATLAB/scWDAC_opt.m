function Zn = scWDAC_opt(X, lambda1, lambda2, maxIter)
%% Normalize data
V = length(X);
for i = 1:size(X,2)
    X{i} = X{i}./(repmat(sqrt(sum(X{i}.^2, 1)),size(X{i}, 1), 1) + eps);
end
[~, N] = size(X{1});

% Calculate D
for v = 1:V
    for i = 1:N  
        for j = 1:N
            D{v}(i,j) = (norm((X{v}(:,i) - X{v}(:,j))))^2;
        end
    end
end

%% ADMM parameters
rho = 2.8;
miu = 0.01;
miu_max = 1e8;

%% Initialization
% Calculate P by KNN with Gaussian kernel function 
for v = 1:V
    k = 10;
    sigma = 1;
    H = construct_knnGraph(X{v}', k);
    P{v} = ones(size(H));
    G = exp(-D{v}/(2*sigma^2));
    for i = 1:length(H)
        IDX = find(H(i,:));
        P{v}(i, IDX) = G(i, IDX);
    end
end
  
CC = zeros(N,N);
for v = 1:V
    C{v} = zeros(N,N);
    Z{v} = zeros(N, N);
    S{v} = zeros(N, N);
    L{v} = zeros(N, N);
    R{v} = zeros(N, N);
    E{v} = zeros(size(X{v}));
    Q1{v} = zeros(size(X{v}));
    Q2{v} = zeros(N, N);
    Q3{v} = zeros(N, N);
    w{v} = 1/V;
end

%% Interation
for iter = 1:maxIter
    %% Update Zv
    for v = 1:V
        M1{v} = X{v} - E{v} + Q1{v}/miu;
        M2{v} = L{v}*C{v}*R{v}' - Q2{v}/miu;
        M3{v} = S{v} - Q3{v}/miu;
        tempZ{v} = X{v}'*X{v} + 2*eye(N);
        Z{v} = tempZ{v}\(X{v}'*M1{v} + M2{v} + M3{v});
    end
    
    %% Update Sv
    for v = 1:V
        S{v} = Z{v} + (Q3{v} - lambda1*P{v}.*D{v})/miu;
        S{v} = S{v} - diag(diag(S{v}));
        for ic = 1:N
            idx = 1:N;
            idx(ic) = [];
            S{v}(ic, idx) = EProjSimplex_new(S{v}(ic, idx));
        end
    end
      
    %% Updata CC 
    Sw = 0;
    SwC = zeros(N, N);
    for v = 1:V
        Sw = Sw + w{v};
        SwC = SwC + w{v}*C{v};
    end
    tempC = SwC/Sw;
    beta = 1/(2*Sw);
    [AC,SC,VC] = svd(tempC, 'econ');
    SC = diag(SC);    
    SVP = length(find(SC>beta));
    if SVP >= 1
        SC = SC(1:SVP) - beta;
    else
        SVP = 1;
        SC = 0;
    end
    CC = AC(:, 1:SVP)*diag(SC)*VC(:, 1:SVP)'; 
    
    %% Update Cv
    for v = 1:V
        N1{v} = Z{v} + Q2{v}/miu;
        C{v} = (miu*L{v}'*N1{v}*R{v} + 2*w{v}*CC)/((miu + 2*w{v})*eye(N));
    end
    
    %% Update Lv and Rv
    for v = 1:V
        tempL{v} = (Z{v} + Q2{v}/miu)*R{v}*C{v}';
        [u, ~, vv] = svd(tempL{v}, 'econ');
        L{v} = u*vv';
        tempR{v} = (Z{v} + Q2{v}/miu)'*L{v}*C{v};
        [u, ~, vv] = svd(tempR{v}, 'econ');
        R{v} = u*vv';
    end
    
    %% Updata Ev
    for v = 1:V
        tempE{v} = X{v} - X{v}*Z{v} + Q1{v}/miu;
        for i = 1:N
            nw = norm(tempE{v}(:,i));
            if nw > lambda2/miu
                x = (nw-lambda2/miu)*tempE{v}(:,i)/nw;
            else
                x = zeros(length(tempE{v}(:,i)),1);
            end
            tempE{v}(:,i) = x;
        end
        E{v} = tempE{v};
    end
    
    %% Updata wv
    for v = 1:V
        w{v} = 1/(2*(norm((C{v}-CC), 'fro')^2 + 10^(-4)))^0.5;
    end
    
    %% Update Q1, Q2, Q3, and miu
    for v = 1:V
        tempQ1{v} = X{v} - X{v}*Z{v} - E{v};
        tempQ2{v} = Z{v} - L{v}*C{v}*R{v}';
        tempQ3{v} = Z{v} - S{v};
        Q1{v} = Q1{v} + miu*tempQ1{v};
        Q2{v} = Q2{v} + miu*tempQ2{v};
        Q3{v} = Q3{v} + miu*tempQ3{v};
    end
    miu = min(rho*miu, miu_max);

    %% Stop
    % Calculate objection value
    for v=1:V
        tempstop = Z{v} - L{v}*C{v}*R{v}';
        temp_ter1(v) = max(max(abs(tempstop)));
    end
    stop = max(temp_ter1);
    error(iter) = stop;
    %------------------------------------------------------------------------
    disp(['iter' num2str(iter) 'stop' num2str(stop)])
    if abs(stop)<1e-2
        break
    end
end
%%
Zn = zeros(N, N);
for v = 1 : V
    Zn = Zn + Z{v};
end
end