function [H_final, W_final,t1,t2,SK1,ZK1] =  LRRNC_RW_TCA(X,k,lambda,beta)
%% RWNMFC
fea = X';
options = [];
options.NeighborMode = 'KNN';
options.k = 5;
options.WeightMode = 'Binary';
W =constructW(fea,options); %传统图正则计算权重矩阵
Ww=computeRRW(W,20);  %随机游走计算概率矩阵代替权重
[~,nSmp]=size(X);
DCol = full(sum(Ww,2));
D = spdiags(DCol,0,nSmp,nSmp);
D = full(D);


Norm = 2;
NormV = 1;


%===========initialization================
[n,m]=size(X);


I = speye(m);
Z0= zeros(m,m);
Zk=Z0; Sk=Z0;
W0= zeros(m,k);
H0= zeros(k,m);
WK=W0; HK=H0;
WK1=WK;HK1=HK;
C_l= zeros(m,m);
C_2= zeros(m,m);
C_1k=C_l;
C_2k=C_2;
iter = 0;
converged = 0;
tol1=1e-5; tol2=1e-5;
flag=0;

mu=10^(-4);
maxIter=100;
t1=zeros(1,maxIter);
t2=zeros(1,maxIter);
%%%%%%%===========Update variables S,Z,W,H by iteration================
while ~converged  && iter < maxIter
    iter = iter + 1;
    flag=0;

    %%%========Update variables Z==========

    a2=Sk-C_l/mu;
    if any(any(isnan(a2))) || any(any(isinf(a2)))
        flag=1;
        disp(['lamuda=' num2str(lambda) ',maxIter=' num2str(maxIter) ',第' num2str(iter) '次，''a2 存在ANY 或者 INF值']);

        break;
    end
   
    ZK1=solve_nn(a2,lambda/mu);

    ZK1=ZK1-diag(diag(ZK1));
    ZK1(ZK1<0)=0;
    %%%========Update variables S==========
    a1=inv(X'*X + I + mu*I);
    SK1=a1*(X'*X +WK*HK + mu*ZK1 + C_l);
    SK1=(SK1+SK1')/2;
    SK1(SK1<0)=0;
    %%%========Update variables W==========
    if any(any(isnan(SK1))) || any(any(isinf(SK1)))
        flag=1;
        disp(['lamuda=' num2str(lambda) ',beta=' num2str(beta) ',mu=' num2str(mu) ',maxIter=' num2str(maxIter) ',第' num2str(iter) '次，''SK1存在ANY 或者 INF值']);
        break;
    end

    if iter==1
        [W0,H0]=nmf(SK1, k, 200);
        [W0,H0] = NormalizeUV(W0, H0', NormV, Norm);H0=H0';
        WK=W0; HK=H0;
    end

    b1=SK1* HK';
    b2=WK*HK*HK';

    WK1=WK.*(b1./b2);
    WK1(WK1<0)=0;
    %%%========Update variables H==========

    a3=WK1'*WK1*HK+2*HK*C_2+beta*HK*D;
    a4=WK1'*SK1+beta*HK*Ww;

    HK1=HK.*(a4./a3);
    HK1(HK1<0)=0;

    %%%========Update variables C_1==========
    C_1k=C_l+mu*(ZK1-SK1);
    C_2k=C_2+mu*(HK1'*HK1-I);


    [WK1,HK1] = NormalizeUV(WK1, HK1', NormV, Norm);HK1=HK1';

    Swk=Sk;
    Zwk=Zk;
    Wwk=WK;
    Hwk=HK;

    Zk=ZK1;
    WK=WK1;
    HK=HK1;
    C_l=C_1k;
    C_2=C_2k;




    temp = max([norm(ZK1-Zwk,'fro'),norm(SK1-Swk,'fro'),norm(WK1-Wwk,'fro'),norm(HK1-Hwk,'fro')]);
    temp =temp/max([norm(X,'fro')]);

    temp1 = max(norm( (X - X*Zk),'fro'),norm( (Zk - WK*HK),'fro'))/max(norm( WK*HK,'fro'),norm( X*Zk,'fro'));
    if temp1 < tol1 && temp < tol2
        converged = 1;
    end

    t1(iter)=temp1;
    t2(iter)=temp;

end %while结束

W_final =WK1; %%% W_final  is finally W
H_final = HK1; %%% H_final  is finally H
[W_final,H_final] = NormalizeUV(W_final, H_final', NormV, Norm);H_final=H_final';

end %found结束


function [ X, s ] = solve_nn( Y, tau )
%% Solves the following
%
%   min tau * |X|_* + 1/2*|X - Y|^2
%
% Created by Stephen Tierney
% stierney@csu.edu.au
%

[U, S, V] = svd(Y, 'econ');
s = diag(S);

ind = find(s <= tau);
s(ind) = 0;

ind = find(s > tau);
s(ind) = s(ind) - tau;

S = diag(s);

X = U*S*(V');

end




function [W, H] = nmf(X, K, MAXITER)
%Euclidean distance
[m,n]= size(X);
% rng(0);
rng(0,'twister')
W = rand(m, K);
H = rand(K, n);
for i=1:MAXITER

    H = H.* (W'*X)./(W'*W*H+eps) ;
    W = W .* (X*H')./(W*H*H'+eps);
end
end

