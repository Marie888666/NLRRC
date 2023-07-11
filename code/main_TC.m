%% jSEG
clear all
clc
addpath(genpath(pwd))
dataset = {'Ramskold','Treutlin','Ting','Goolam','Deng','mECS','Engel','Pollen','Darmanis','Kolod','Tasic','Zeisel','Quake_10x_Limb_Muscle'};

be=[10^(-1),10^(-1),10^(-1),10^(-1),10^(-4),10^(-5),10^(-4),10^(-5),10^(-5),10^(-1),10^(-2),10^(-2),10^(-1)];
lambda = 10^(-5);
result= zeros(2,13);
j=1;
for num =8:8
    load(['Data_' dataset{num}]);
    if num ==13
        load([dataset{num}]);
    end

    in_X = double(in_X);
    label=true_labs';
    n_space = length(unique(label));
    K = n_space;
    [X,] = FilterGenesZero(in_X);
    [n,m]=size(X);
    X = normalize(X');


    beta=be(num);
    [H,W,t1,t2,SK1,ZK1]= LRRNC_RW_TCA(X,K,lambda,beta);


    %%%%%%%%%%% Clustering cell type label
    ll=label;
    [~,indx] = max(abs(H));
    l=indx;

    [~,a1]=size(ll);
    [~,a2]=size(l);
    if a1~=1
        ll=ll';
    end
    if a2~=1
        l=l';
    end

    NMI=Cal_NMI(ll, l);
    ARI=Cal_ARI(ll, l);
    result(1,j)=NMI;
    result(2,j)=ARI;
    j=j+1;

end






