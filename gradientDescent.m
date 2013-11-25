function [ Pnew ] = gradientDescent( X, Y, P, H )

    [N,C] = size(P);
    Pnew = zeros(N,C);
    alpha = 0.0001;
    
    
    for i = 1:N
        cost = alpha * (Y(i,1) - H(i,:));
        descent = cost' * X(i,:);
        
        disp(size(P));
        disp(size(descent));
        Pnew(i,:) = P(i,:) - descent;
    end
    

end

