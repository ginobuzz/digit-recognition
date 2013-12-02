function W = train_lr()
%==========================================================================
% train_lr: Trains learning parameter matrix using the SoftMax function.
%
%   Output:
%       W - DxK sized learning parameter matrix.
%
%   Author: ginobuzz
%==========================================================================


    %===============================================
    % Constants
    %-----------------------------------------------
    LearnRate = 0.0001;              % Learning Rate
    Convergence = 0.0005;     % Convergence Criteria
    MaxIterations = 1000;       % Maximum Iterations
    K = 10;                      % Number of classes
    %===============================================

      
    % Load training data.
    F = load('Train.mat');
    X = F.X;
    T = F.T;
    
    % Append column of ones for bias
    X = horzcat(ones(length(X),1),X);

    % Determine size of training data.
    [N,D] = size(X);
    
    % Initialize DxK matrix of learning parameters to random values 
    % ranging from 0.0 to 0.5.
    W = randi([0,5],D,K) / 10;
       
    % Initialize error value.
    error = 0;
    
    numIterations = 0;
    while numIterations < MaxIterations;
        
        % Create Activation Matrix.
        A = X * W;
        
        % Create Prediction Matrix using SoftMax function.
        Y = zeros(N,K);
        for n = 1:N
            Ak = exp(A(n,:));
            AkSum = sum(Ak);
            for k = 1:K
                Y(n,k) = Ak(1,k) / AkSum;
            end
        end
        
        % Calculate error of predictions. 
        newError  = 0;
        for n = 1:N
            for k = 1:K
                newError = newError + (T(n,k) * log(Y(n,k)));
            end
        end
        newError = -newError/N;
        
        % Check for convergence.
        if abs(error - newError) < Convergence
            error = newError;
            break;
        end
        error = newError;
        
        % Calculate gradient and adjust learning parameters.
        gradient = LearnRate * X' * (Y - T);
        W = W - gradient;
        
        numIterations = numIterations + 1;
    end
    
    fprintf('[Logistic Regression] Training Error: %f \n', error);
    
end
