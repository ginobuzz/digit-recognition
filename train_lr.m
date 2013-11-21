function [W] = train_lr()

    % Regularization coefficient
    LAMBDA = 0.2;

    % Get train and validation sets.
    [T,V] = formatData();
    
    % Apply sigmoid function to each matrix element.
    X = T{1};
    [M,N] = size(X);
    
    for i = 1:M
        for j = 2:N
            X(i,j) = sigmoid(X(i,j));
        end
    end
    
    
    W = inv((X'*X) + (LAMBDA*eye(N))) * (X'*V{1});
    
    function [phi] = sigmoid(x)
        phi = 1 / (1 + exp(-x));
    end

end
