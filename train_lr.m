function [W, trainData, testData] = train_lr()
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
    LearnRate = 0.01;              % Learning Rate
    Convergence = 0.00018;     % Convergence Criteria
    MaxIterations = 200;       % Maximum Iterations
    K = 10;                      % Number of classes
    %===============================================

    trainData = zeros(1000,1);
    testData = zeros(1000,1);
    
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
    %W = randi([0,5],D,K);
    W = zeros(D,K);
    
    % Initialize error value.
    error = 0;
    oldDif = 0;
    for i = 1: MaxIterations
        
        % Create Activation Matrix.
        A = X * W;
        
        % Create Prediction Matrix using SoftMax function.
        Y = zeros(N,K);
        
        Ak = exp(A);
        AkSum = zeros(N,1);
        for n = 1:N
            AkSum(n,1) = sum(Ak(n,:));
        end
        for n = 1:N
            Y(n,:) = Ak(n,:) ./ AkSum(n,1);
        end
        
        
        % Calculate error of predictions. 
        newError = 0;
        for k = 1:K
            cost = T(:,k)' * log(Y(:,k));
            newError = newError + cost;
        end
        newError = -newError/N;
        
        fprintf('Training Error: %f \n', newError);
        
        % Check for convergence.
        newDif = error - newError;
        if abs(error - newError) < Convergence
            error = newError;
            fprintf('Convergence criteria met.\n');
            break;
        end
        error = newError;
       
        
        %LearnRate = LearnRate * 1.0001;
        
        if(mod(i, 5) == 0)
            %LearnRate = LearnRate * 0.95;
        end
        if((error - newError ) < 0.005)
           % LearnRate = LearnRate * 1.015;
        end
           
        if newDif < oldDif
            LearnRate = LearnRate * 1.04;
        else
            LearnRate = LearnRate * 0.96;
            oldDif = newDif;
        end
        
        
        % Calculate gradient and adjust learning parameters.
        gradient = (LearnRate/N) * X' * (Y - T);
        W = W - gradient;
        
        
        % Form boolean prediciton matrix to compare to T.
        P = zeros(N,K);
        for n = 1:N
            [C,I]  = max(Y(n,:));
            P(n,I) = 1;
        end
        numIncorrect = 0;
        for n = 1:N
            if ~isequal(P(n,:),T(n,:))
                numIncorrect = numIncorrect + 1;
            end
        end
        
        % Calculate final error.
        te = numIncorrect / N;
        
        trainData(i,1) = te;
        testData(i,1) = test_lr(W);
        
    end
    
    fprintf('[Logistic Regression] Training Error: %f \n', error);
    
end
