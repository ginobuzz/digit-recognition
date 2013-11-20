function [X] = train_lr()
%TRAIN_LR Summary of this function goes here
%   Detailed explanation goes here

    % Import Training Sets
    D = load('features_train/0.txt');
    [M,N] = size(D);
    X = zeros(M,N+1);
    
    for i = 1:M
        label = i -1;
        X(i,:) = [label D(i,:)];
    end

end