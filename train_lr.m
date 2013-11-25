function [Theta] = train_lr()
%==========================================================================
% train_lr: Linear Regression Classifier
%
%   Input:
%       none
%
%   Output:
%       P - Parameter Matrix.
%
%   Author: ginobuzz
%==========================================================================

    % Error threshold; Convergence criteria.
    E = 0.1;
    
    
    [X, Y] = formatData();  % Initialize training matrix and labels.
    P = rand(10, 513);      % Initialize parameter matrix.

    
    error = 100;
    while error > E
        H = hypothesis_lr(X,Y,P); % Build hypothesis matrix.
        P = gradientDescent(X,Y,P,H);
        error = 0.01;
    end
    
    
    



    function c = cost( H )
        arg1 = (-y) * log(h);
        arg2 = (1 - y) * log(1 - h);
        c = arg1 - arg2;
    end
    
end



