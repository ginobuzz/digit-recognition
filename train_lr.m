function W = train_lr()
%==========================================================================
% train_lr: Linear Regression Classifier
%
%   Input:
%       none
%
%   Output:
%       W - Parameter Matrix.
%
%   Author: ginobuzz
%==========================================================================

    % Convergence criteria
    E = 0.01;

    % Number of classes.
    M = 10


    % Initialize training matrix, boolean target matrix, and label vector. 
    [X,T,L] = formatData('features_train/');
    [N,D]   = size(X);

    % Initialize parameter matrix.
    W = zeros(D,M);

    error = 100;
    while error > E

        % Build hypothesis matrix.
        Y = hypothesis(X,W);

        % Get new parameter matrix, using gradient descent.
        W_new = gradientDescent(X,T,W,Y);

        % Update error.
        %! CHANGE 
        error = 0;
    end

end
