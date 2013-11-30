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

    % Start clock
    tic;

    % Learning Rate
    Alpha = 0.0001;

    % Convergence criteria
    E = 0.0005;

    % Number of classes.
    M = 10;

    % Initialize training matrix, boolean target matrix, and label vector. 
    [X,T,L] = formatData('features_train/');
    [N,D] = size(X);
    
    % Initialize parameter matrix (W).
    W = randi([0,5], D, 10) / 10;
    
    prevError = 100;
    error = 0;
    
    while abs(prevError - error) > E
        
        % Build activation matrix (A).
        A = X * W;

        % Build hypothesis matrix (Y).
        Y = zeros(N,10);
        for n = 1:N
            expAk = exp(A(n,:));
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
        %fprintf('Error: %f \n', error);
        
        % Calculate gradient
        gradient = Alpha * X' * (Y - T);
        W = W - gradient;
        
    end
    
    fprintf('Learning parameters trained. [Operation took %f seconds.]\n',toc); 
    
end
