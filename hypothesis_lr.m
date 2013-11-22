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
%       H - Hypothesis Matrix of size Mx513.
%
%   Author: ginobuzz
%==========================================================================

    [N,D] = size(X);
    H = zeros(N,513);
    
    for n = 1:N
        
        pIndex = Y(n,1) + 1;
        p = P(pIndex,:)';
        
        for d = 1:D
            px = p(d,1) * X(n,d);
            H(n,d) = sigmoid(px);
        end
        
    end
   
    
% Local Functions
%==========================================================================
    

    function s = sigmoid( a )
        s = 1 / ( 1 + exp(-a) ); 
    end


end

