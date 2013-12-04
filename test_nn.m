function [error, Y] = test_nn(W_hidden, W_output)


    %===============================================
    % Constants
    %-----------------------------------------------
    K = 10;                      % Number of classes
    %===============================================
    

    % Load training data.
    F = load('Test.mat');
    T = F.T;
    [N,M] = size(F.X);
    
    % Forward Propagate: input-layer --> hidden-layer.
    % ---------------------------------------------------   
    X = horzcat(ones(length(F.X),1), F.X);
    A = X * W_hidden;
    Z = tanh(A);
         
    % Forward Propagate: hidden-layer --> output-layer.
    % ---------------------------------------------------
    Z = horzcat(ones(length(Z),1),Z);
    A = Z * W_output;
    Y = zeros(N,K);
    Ak = exp(A);
    AkSum = sum(Ak,2);
    for k = 1:K
        Y(:,k) = Ak(:,k) / AkSum(k,1);
    end
    
    % Derive boolean prediction matrix from Y.
    % ---------------------------------------------------
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

