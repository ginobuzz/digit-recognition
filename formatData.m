function [ X, Y ] = formatData()
%=========================================================================
% formatData: function combines all 10 training data files into a single
%   cell-array where each element of the cell-array is a matrix.
% 
%   Input: 
%       none
%
%   Output:
%   	X - Training Matrix of size NxD.
%       Y - Label Vector of size Nx1.
%     
%   Author: ginobuzz
%=========================================================================

    DIR = 'features_train/'; % Directory path.
    EXT = '.txt';            % File extension.
    
    fprintf('> Starting Data Format...\n');
    tic;
   
    T = [];
    for i = 1:10
        
        index = i - 1;
        file_path = strcat(DIR, num2str(index), EXT);
        try
            D = load(file_path);
        catch
            eMsg = strcat('> [ERROR] File not found (', file_path, ').');
            error(eMsg);
        end
        
        [r,c] = size(D);
        bias  = ones(r,1);
        label = bias * (i-1);
        tmp = [label bias D];
        T = [T; tmp];
        
    end
    
    Y = T(:,1);
    X = T(:,2:514);
    
    fprintf('> Data Format: Successful. [Operation took %f seconds]\n', toc);

end

