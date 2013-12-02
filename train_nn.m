function [Y,T,Z,error, W1, W2] = train_nn()

    ErrorThreshold = 2;
    LR = 0.001;

    % X - NxD Input matrix.
    % T - NxK Target matrix.
    
    % K - Number of classes.
    K = 10;
    
    F = load('Train.mat');
    X = F.X; [N,D] = size(X); 
    T = F.T;
    
    % Number of neurons in hidden layer.
    S = 1000;
    
    % Create random weight matrix.
    W1 = randi([1,9], D, S) / 10;
    W2 = randi([1,9], S+1, K) / 10;
    
    error = 100;
    count = 1
    while count < 10
        count = count + 1;
        
        % Forward-propagate from input-layer to hidden-layer.
        A = X * W1;
        Z = horzcat(ones(N,1), tanh(A));

        % Forward-propagate from hidden-layer to output-layer.
        A = Z * W2;
        Y = zeros(N,K);
        for n = 1:N
            Ak = exp(A(n,:));
            AkSum = sum(Ak);
            for k = 1:10
                Y(n,k) = Ak(1,k) / AkSum;
            end
        end    

        error = calculateError(T,Y);
        if error < ErrorThreshold
            disp('Threshold Met');
            break;
        end

        % Calculate error of output layer.
        E2 = Y - T;
        % Calculate output layer gradient.
        G2 = Z' * E2;

        % Calculate error of hidden layer.
        E1 = (W2 * E2')' .* (1 - Z.^2); 
        % Calculate gradient of hidden layer.
        G1 = X' * E1;

        % Adjust parameters.
        W2 = W2 - ((LR/N) * G2);
        W1 = W1 - ((LR/N) * G1(:,2:end));
    
    end
    
end

function error = calculateError(T,Y)
    [N,K] = size(Y);
    error = 0;
    for k = 1:K
        v1 = T(:,k)' * log(Y(:,k));
        v2 = (1 - T(:,k))' * log(Y(:,k));
        cost  = v1 + v2;
        error = error + cost;
    end
    error = -1/N * error;
end



function Z = FP_Hidden(X, W)
%==========================================================================
% Input:
%   X - Input Neurons.
%   W - Learning Parameters.
%   actFunc - Activation function handle.
%==========================================================================
    [N,D] = size(X);
    A = X * W;
    Z = horzcat(ones(N,1), tanh(A));
end













