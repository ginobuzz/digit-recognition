function [W_layer1, W_layer2] = train_nn()
%==========================================================================
% train_nn: Trains learning parameter matrix using a Neural Network.
%
%   Output:
%       W - DxK sized learning parameter matrix.
%
%   Author: ginobuzz
%==========================================================================

    % Learning Rate
    LearningRate = 0.1;
    
    % Convergence Criteria
    Convergence = 0.000002;

    % Number of Hidden Units
    H = 360;

    % Number of Max Iterations
    MaxIterations = 20;

    % Number of Classes
    K = 10;

%--------------------------------------------------------------------------

    % Load training data.
    F = load('Train.mat');
    T = F.T;
    X = horzcat(ones(length(F.X)), F.X);
    [N,D] = size(X);
    
    W_layer1 = randi([-10000,10000], D, H)/10000;
    W_layer2 = randi([-10000,10000], H+1, K)/(10000 * N);

    prevError = 0;
    for i = 1:MaxIterations

        
        [Z, Y] = forwardPropagation(X, W_layer1, W_layer2, N);
        
        [E, G_layer1, G_layer2] = backPropagation(X, Z, W_layer2, Y, T, N);
        
        disp(size(W_layer1));
        disp(size(G_layer1));
        W_layer1 = W_layer1 - (LearningRate/N * G_layer1(:, 2:end));
        
        W_layer2 = W_layer2 - (LearningRate/N * G_layer2);
        
        
%         % Forward Propagate: input-layer -> hidden-layer.
%         Z = forwardProp1(X, W_layer1, N);
% 
%         % Forward Propagate: hidden-layer -> output-layer.
%         Y = forwardProp2(Z, W_layer2, N);
% 
%         % Back Propagate: hidden-layer <- output-layout.
%         [trainError, E_layer2, G_layer2] = backProp2(Y, T, Z, N);
%         
%         % Back Propagate: input-layer <- hidden-layout.
%         [E_layer1, G_layer1, G_layer2] = backProp1(X, Z, W_layer2, E_layer2);
%         
% 
%         % Check for convergence.
%         if abs(prevError - trainError) < Convergence
%             error = trainError;
%             fprintf('Convergence criteria met.\n');
%             break;
%         end
%         error = trainError;
%         
%         LearningRate = LearningRate / 1.03;
%         
%         % Gradient Descent.
%         disp(size(W_layer1));
%         disp(size(G_layer1));
%         
%         W_layer1 = W_layer1 - (LearningRate/N * G_layer1(:, 2:end));
%         W_layer2 = W_layer2 - (LearningRate/N * G_layer2);

    end

    fprintf('Number of Iterations: %d\n', curIteration);
    fprintf('Final Training Error: %f \n', E);
    
end


function [Z,Y] = forwardPropagation(X, W1, W2, N)
    A = X * W1;
    Z = horzcat(ones(N,1),tanh(A));
    Y = zeros(N,10);
    A = exp(Z * W2);
    ASum = sum(A,2);
    for k = 1:10
        Y(:,k) = A(:,k) / ASum(k,1);
    end
end

function [error, G_layer1, G_layer2] = backPropagation(X, Z, W2, Y, T, N)
    error = 0;
    for k = 1:10
        error = error + (T(:,k)' * log(Y(:,k)));
    end
    error = -error/N;
    fprintf('Training Error: %f \n', error);
    
    delta_layer2 = Y - T;
    G_layer2 = Z' * delta_layer2;
    
    delta_layer1 = (W2 * delta_layer2')' .* (1 - (Z.^2));
    G_layer1 = X'* delta_layer1;
end




