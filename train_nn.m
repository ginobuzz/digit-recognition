function [W_L1, W_L2, plotData, lrData, testData] = train_nn()
%==========================================================================
% train_nn: Trains learning parameter matrix using a Neural Network.
%
%   Output:
%       W - DxK sized learning parameter matrix.
%
%   Author: ginobuzz
%==========================================================================

    % Learning Rate
    LR = 0.5;
    LRmax = 0.99;
    LRmin = 0.01;
    stepUp = 1.15;
    stepDown = 0.55;
    
    
    % Convergence Criteria
    Convergence = 0.00001;

    % Number of Hidden Units
    H = 350;

    % Number of Max Iterations
    MaxIterations = 200;

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
    W = load('FixedWeights.mat');
    %W_L1 = W.W_L1;
    %W_L2 = W.W_L2;
    
    prevError = 100;
    oldDif = 0;

    plotData = zeros(MaxIterations,1);
    lrData = zeros(MaxIterations,1);
    testData = zeros(MaxIterations,1);
    
    for i = 1:MaxIterations
        
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
        
        error = 0;
        for k = 1:K
            error = error + (T(:,k)' * log(Y(:,k)));
        end
        error = -(1/N) * error;
        fprintf('Error: %f. (I: %d) (LR: %f) \n', error, i, LR);
 
        
        errorChange = prevError - error;
        newDif = errorChange;
        
        if abs(errorChange) < Convergence
            if error < 1.5
            
                fprintf('Convergence Criteria Met\n');
                prevError = error;
                break;
            end
        end
        prevError = error;
        
        
        
        
        DeltaL2 = Y - T;
        GradientL2 = Z' * DeltaL2;
        
        DeltaL1 = (1 - Z.^2) .* (W_L2 * DeltaL2')';
        GradientL1 =  X' * DeltaL1;
        
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
        te = numIncorrect / N;
        
        LRmax = te * 0.7;
        
        if newDif < oldDif
            LR = min(LR*stepUp, LRmax);
        elseif newDif == 0
            LR = min(LR*stepUp, LRmax);
        else
            %LR = 0.5;
            LR = max(LR*stepDown, LRmin);
            %oldDif = max(newDif, 0);
        end 
        oldDif = max(newDif, 0);
        
        
        W_L1 = W_L1 - ((LR/N) * GradientL1(:,2:end));
        W_L2 = W_L2 - ((LR/N) * GradientL2);
        
        
        
        plotData(i,1) = te;
        lrData(i,1) = LR;
        testData(i,1) = test_nn(W_L1, W_L2);
        
        
    end
    
    fprintf('Final Training Error: %f \n', prevError);
    
end
