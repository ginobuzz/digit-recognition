function [W_new, error] = gradientDescent( X, T, W, Y )

    W_new = zeros(size(W));
    [N,D] = size(X);
    
    gradient = zeros(10,D);
    
    for m = 1:10
        
        for n = 1:N
            cost     = (Y(n,m) - T(n,m)) * X(n,:);
            gradient(m,:) = gradient(m,:) + (cost);
        end
        
        W_new(:,m) = W(:,m) - gradient(m,1)';
        
    end
    
    error = sum(gradient);

end
