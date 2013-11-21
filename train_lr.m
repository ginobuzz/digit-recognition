function [Y] = train_lr()


    % Get train and validation sets.
    [Train, Valid] = formatData();
    [M,N] = size(Train);
    X = [ones(M,1) Train(:,2:N)]; % Data Matrix
    Y = Train(:,1);               % Target Vector
    
    % Create Parameter matrix.
    W = rand(N,10);
    
    % Create Hypothesis matrix.
    H = hypothesis(X,W);
    
    
    
    
    
    
    
    function [H] = hypothesis( X, W )
        [r,c] = size(X);
        H = zeros(r,c);
        
        for i = 1:r
            z = 
        end
        
    end




    function c = cost( h, y )
        arg1 = (-y) * log(h);
        arg2 = (1 - y) * log(1 - h);
        c = arg1 - arg2;
    end
    
    


end
