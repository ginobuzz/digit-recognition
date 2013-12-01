function [W] = train_lr()
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

    Alpha = 0.0001;% Learning Rate.
    E = 0.0005;% Convergence Criteria.
    M = 10;% Number of classes.

    % Start clock
    tic;

    % Initialize training matrix, boolean target matrix, and label vector. 
    [X,T,L] = formatData('features_train/');
    [N,D] = size(X);
    
    % Initialize parameter matrix (W).
    W = randi([0,5], D, 10) / 10;
    
    prevError = 100;
    error = 0;
    
    while abs(prevError - error) > E
        
        A = X * W;% Activation Matrix.
        Y = zeros(N,10);% Hypothesis Matrix.
        
        for n = 1:N
            expAk  = exp(A(n,:));
            expSum = sum(expAk);
            for k = 1:10
                Y(n,k) = expAk(1,k) / expSum;
            end
        end
        
        % Calculate error
        prevError = error;
        error = 0;
        for n = 1:N
            for k = 1:10
                error = error + (T(n,k) * log(Y(n,k)));
            end
        end
        error = -error / N;
        
        % Calculate gradient
        gradient = Alpha * X' * (Y - T);
        W = W - gradient;
        
    end
    
    fprintf('Learning parameters trained. [Operation took %f seconds.]\n',toc); 
    
end
