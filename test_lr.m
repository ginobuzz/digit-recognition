function error = test_lr(W)
%==========================================================================
% test_lr: Tests learning parameters against the test feature set.
%
%   Input:
%       W - DxK sized learning parameter matrix.
%
%   Output:
%       error - test error.
%
%   Author: ginobuzz
%==========================================================================
   

    %===============================================
    % Constants
    %-----------------------------------------------
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
    error = numIncorrect / N;
    fprintf('[Logistic Regression] Test Error = %f \n',error);
    
end

