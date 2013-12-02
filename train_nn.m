function [W_hidden, W_output] = train_nn()
%==========================================================================
% train_nn: Trains learning parameter matrix using a Neural Network.
%
%   Output:
%       W - DxK sized learning parameter matrix.
%
%   Author: ginobuzz
%==========================================================================


    %===============================================
    % Constants
    %-----------------------------------------------
    LearnRate = 0.0015;              % Learning Rate
    Convergence = 0.005;      % Convergence Criteria
    MaxIterations = 100;       % Maximum Iterations
    K = 10;                      % Number of classes
    S = 1000;                 % Size of hidden layer
    %===============================================

    
    % Load training data.
    F = load('Train.mat');
    T = F.T;
    
    % Get size of data.
    [N,M] = size(F.X);
    
    % Initialize learning parameter matrix for each layer.
    W_hidden = randi([1,9], M+1, S)/10;
    W_output = randi([1,9], S+1, K)/10;

    % Initialize error value.
    error = 0;
    
    numIterations = 0;
    while numIterations < MaxIterations
    
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
        
      % Calculate Error
      % ---------------------------------------------------
        newError = 0;
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

      % Back Propagate: hidden-layer <-- output-layout.
      % ---------------------------------------------------
        % Calculate gradient.
        delta_output = Y - T;
        gradient_output = Z' * delta_output;
        
        
      % Back Propagate: input-layer <-- hidden-layout.
      % ---------------------------------------------------
        % Calculate gradient.
        delta_hidden = (W_output * delta_output')' .* (1 - (Z.^2));
        gradient_hidden = X' * delta_hidden;
        
        
      % Gradient Descent.
      % ---------------------------------------------------
        W_hidden = W_hidden - (LearnRate/N * gradient_hidden(:, 2:end));
        W_output = W_output - (LearnRate/N * gradient_output);
        numIterations = numIterations + 1;
    end

    fprintf('[Neural-Network] Training Error: %f \n', error);
end













