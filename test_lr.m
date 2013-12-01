function Error = test_lr(W)
%==========================================================================
% test_lr: Tests learning parameters.
%
%   Input:
%       W - Learning Parameter Matrix.
%
%   Output:
%       error - Percentage of incorrectly classified labels.
%
%   Author: ginobuzz
%==========================================================================
    [X,T,L] = formatData('features_test/');
    [N,D]   = size(X);

    % Build activation matrix (A).
    A = zeros(N,10);
    for k = 1:10
        A(:,k) = X * W(:,k);
    end

    % Build hypothesis matrix (Y).
    Y = zeros(N,10);
    for n = 1:N
        expSum = 0;
        for k = 1:10
            expSum = expSum + exp(A(n,k));
        end

        for k = 1:10
            Y(n,k) = exp(A(n,k)) / expSum;
        end
    end
    
    
    % Form predictions
    P = zeros(N,1);
    for n = 1:N
        p = 0;
        for k = 1:10
            if Y(n,k) > p
                p = Y(n,k);
                P(n,1) = k - 1;
            end
        end
    end
    
    % Compare
    numIncorrect = 0;
    for n = 1:N
        if P(n,1) ~= L(n,1)
            numIncorrect = numIncorrect + 1;
        end
    end
    
    Error = numIncorrect / N;
    
end

