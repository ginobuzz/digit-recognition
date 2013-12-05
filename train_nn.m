function [W_L1, W_L2, Y] = train_nn()
%==========================================================================
% train_nn: Trains learning parameter matrix using a Neural Network.
%
%   Output:
%       W - DxK sized learning parameter matrix.
%
%   Author: ginobuzz
%==========================================================================

    % Learning Rate
    LR = 0.2;
    
    % Convergence Criteria
    Convergence = 0.0000000001;

    % Number of Hidden Units
    H = 350;

    % Number of Max Iterations
    MaxIterations = 2000;

    % Number of Classes
    K = 10;

%--------------------------------------------------------------------------

    % Load training data.
    F = load('Train.mat');
    X = horzcat(ones(length(F.X), 1), F.X);
    T = F.T;
    [N,D] = size(X);

    % Initialize Weights.
    W_L1 = randi([-10,10], D, H) / (10);
    W_L2 = randi([-10,10], H+1, K) / (10 * N);
    %W = load('FixedWeights.mat');
    %W_L1 = W.W_L1;
    %W_L2 = W.W_L2;
    
    prevError = 100;
    
    prevDelta1 = zeros(1, H+1);
    prevDelta2 = zeros(1, K);
    
    
    for i = 1:MaxIterations
        
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

        error = 0;
        for k = 1:K
            error = error + (T(:,k)' * log(Y(:,k)));
        end
        error = -(1/N) * error;
        fprintf('Training Error: %f \n', error);
 
        
        errorChange = prevError - error;
        if abs(errorChange) < Convergence
            fprintf('Convergence Criteria Met\n');
            prevError = error;
            break;
        end
        prevError = error;
        
        
        DeltaL2 = Y - T;
        [GradientL2, prevDelta2] = RPropGradient(Z, DeltaL2, prevDelta2);
        
        DeltaL1 = (1 - Z.^2) .* (W_L2 * DeltaL2')';
        GradientL1 = X' * DeltaL1; 
        %[GradientL1, prevDelta1] =  RPropGradient(X, DeltaL1, prevDelta1);
        
        W_L1 = W_L1 - (0.5 * GradientL1(:,2:end));
        W_L2 = W_L2 - GradientL2;
        
        
        
        
    end
    
    fprintf('Final Training Error: %f \n', prevError);
    
end

function [Gradient, D] = RPropGradient(Neurons, Delta, prevD)
    Gradient = Neurons' * Delta;
    [N,M] = size(Delta);
    D = sum(Delta);
    diff = prevD - D;
    for m = 1:M
        if diff(1,m) > 0
            Gradient(:,m) = 0.5 * Gradient(:,m);
        else
            Gradient(:,m) = 1.2 * Gradient(:,m);
        end
    end
    
end





