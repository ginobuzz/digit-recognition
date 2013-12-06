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
    F = load('Test.mat');
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
    Ak = exp(A);
    AkSum = zeros(N,1);
    for n = 1:N
        AkSum(n,1) = sum(Ak(n,:));
    end
    for n = 1:N
        Y(n,:) = Ak(n,:) ./ AkSum(n,1);
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
    %fprintf('Test Error = %f \n',error);
    
end

