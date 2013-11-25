function [W_new, error] = gradientDescent( X, T, W, Y )

    ALPHA = 0.0001;

    % Initialize new W matrix.
    [D,M] = size(W);
    W_new = zeros(D,M);

    error = 0;

    for i = 1:D
    
        diff    = Y(i,:) - T(i,:);
        cost    = diff' * X(i,:);
        descent = ALPHA * cost;  

        error = error + cost;

        W_new(i,:) = W(i,:) - descent;
    
    end

end
