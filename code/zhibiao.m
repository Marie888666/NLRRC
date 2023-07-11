function [F] = zhibiao(Rnk) 
%    Unitization
     [a,b] = size(Rnk);
     for i =1:b
       A (1,i)= norm(Rnk(:,i));
     end
     A = repmat(A,a,1);
     F= Rnk./A; 
end