function [error] = test_nn(W_L1, W_L2)


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
    Ak = exp(A2);
    AkSum = zeros(N,1);
    for n = 1:N
        AkSum(n,1) = sum(Ak(n,:));
    end
    for n = 1:N
        Y(n,:) = Ak(n,:) ./ AkSum(n,1);
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
    
    
    %fprintf('Final Test Error = %f \n',error);
    
end

