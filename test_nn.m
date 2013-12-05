function [error, Y] = test_nn(W_L1, W_L2)


    %===============================================
    % Constants
    %-----------------------------------------------
    K = 10;                      % Number of classes
    %===============================================
    
    
    % Load training data.
    F = load('Test.mat');
    X = horzcat(ones(length(F.X), 1), F.X);
    T = F.T;
    [N,D] = size(X);
    
    A1 = X * W_L1;
    Z  = horzcat(ones(N,1),tanh(A1));
    
    Y = zeros(N,10);
    A2 = Z * W_L2;
    A2 = 1 ./ (1 + exp(-A2));
    A2 = exp(A2);
    A2Sum = sum(A2,2);
    for k = 1:K
        Y(:,k) = A2(:,k) / A2Sum(k,1);
    end

    P = zeros(size(T));
    for n = 1:N
        [C,I] = max(Y(n,:));
        P(n,I) = 1;
    end
    
    numIncorrect = 0;
    for n = 1:N
        if ~isequal(P(n,:),T(n,:))
            numIncorrect = numIncorrect + 1;
        end
    end
    error = numIncorrect/N;
    
    
    fprintf('Final Test Error = %f \n',error);
    
end

