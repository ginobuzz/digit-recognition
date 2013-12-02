function [error, Y] = test_nn(W_hidden, W_output)


    %===============================================
    % Constants
    %-----------------------------------------------
    K = 10;                      % Number of classes
    %===============================================
    

    % Load training data.
    F = load('Train.mat');
    T = F.T;
    
    % Get size of data.
    [N,M] = size(F.X);
    
    
    % Forward Propagate: input-layer --> hidden-layer.
    % ---------------------------------------------------    
    % Add column of ones for bias.
    X = horzcat(ones(length(F.X),1), F.X);
    % Apply activation function (tanh).
    A = X * W_hidden;
    Z = tanh(A);
         
    % Forward Propagate: hidden-layer --> output-layer.
    % ---------------------------------------------------
    % Add column of ones for bias.
    Z = horzcat(ones(length(Z),1),Z);
    % Apply activation function (softmax).
    A = Z * W_output;
    Y = zeros(N,K);
    for n = 1:N
        Ak = exp(A(n,:));
        AkSum = sum(Ak);
        for k = 1:K
            Y(n,k) = Ak(1,k) / AkSum;
        end
    end
    
    % Derive boolean prediction matrix from Y.
    % ---------------------------------------------------
    P = zeros(size(T));
    for n = 1:N
        [C,I] = max(Y(n,:));
        P(n,I) = 1;
    end
    
    % Calculate Error
    % ---------------------------------------------------
    numError = 0;
    for n = 1:N
        if ~isequal(P(n,:), Y(n,:))
            numError = numError + 1;
        end
    end
    error = numError/N;
    fprintf('[Neural Network] Test Error = %f \n',error);
    
end

