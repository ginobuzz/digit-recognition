function [Y] = train_lr()


    % Get train and validation sets.
    [T,V] = formatData();

    tmp = T{1};
    [M,N] = size(tmp);
    
    mag = magic(N);
    W = mag(:,1);
    
    X = [ones(M,1) tmp(:, 2:N)];
    Y = zeros(M,N);
    
    for i = 1:M
        Y(i,:) = sigmoid( X(i,:), W );
    end
    
    
    
    
    
    function phi = sigmoid( x, w )
        a = x*w;
        phi = 1 / (1 + exp(-a));
    end
   


end
