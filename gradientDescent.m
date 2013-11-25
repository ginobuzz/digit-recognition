function W_new = gradientDescent( X, T, W, Y )

    ALPHA = 0.0001;


    % Initialize new W matrix.
    [D,M] = size(W);
    W_new = zeros(D,M);

    for i = 1:D
    
        cost    = Y(i,:) - T(i,:);
        regCost = ALPHA * cost;
        descent = cost' * X(i,:);  

        W_new(i,:) = W(i,:) - descent;
    
    end

end

