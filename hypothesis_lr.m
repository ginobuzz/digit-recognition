function [ H ] = hypothesis_lr( X, Y, P )
%==========================================================================
% hypothesis_lr: Builds the hypothesis matrix.
%
%   Input:
%       X - Training Matrix of size Mx513.
%       Y - Label Vector of size Mx1.
%       P - Parameter Matrix of size 10x513.
%
%   Output:
%       H - Hypothesis Matrix of size Nx10, 
%           where H(i,j) = P( Cj | X(i,:) ). 
%
%   Author: ginobuzz
%==========================================================================

    [N,D] = size(X);
    H = zeros(N,10);
    
    for n = 1:N
        a = zeros(1,10);
        
        for c = 1:10
            a(1,c) = exp(dot(P(c,:), X(n,:)));
        end
        
        for c = 1:10
           ak = a(1,c);
           aj = sum(a) - ak;
           H(n,c) = ak/aj; 
        end
    end
end

