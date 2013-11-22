function [P] = train_lr()
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

    % Get train and validation sets.
    [X, Y] = formatData();
    
    % Create Parameter matrix.
    P = rand(10, 513);

    % Build hypothesis matrix.
    H = hypothesis_lr(X, Y, P);
    
    % Perform gradient decent.
    



    function c = cost( H )
        arg1 = (-y) * log(h);
        arg2 = (1 - y) * log(1 - h);
        c = arg1 - arg2;
    end
    
end



