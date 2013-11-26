function Y = hypothesis_lr( X, W )
%==========================================================================
% hypothesis_lr: Builds the hypothesis matrix.
%
%   Input:
%       X - Training Matrix of size Mx513.
%       W - Parameter Matrix of size 513x10.
%
%   Output:
%       Y - Hypothesis Matrix of size Nx10.
%
%   Author: ginobuzz
%==========================================================================

    [N,D] = size(X);
    Y = zeros(N,10);
    
    for n = 1:N
       
        e = zeros(10,1);
        for m = 1:10
            e(m,1) = exp(dot(W(:,m)',X(n,:)));
        end 
        
        eSum = sum(e);
        
        for m = 1:10
            Y(n,m) = e(m,1) / eSum;
        end
        
    end

    
end

