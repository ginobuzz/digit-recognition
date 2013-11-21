function [ T, V ] = formatData()
%=========================================================================
% formatData: function converts all 10 training-data sets into two 
%   labeled matracies; one for training and one for validation.
% 
% Input: 
%   none
%
% Output:
%   T - Mx513 sized labeled training matrix. 
%   V - Dx513 sized labeled validation matrix.
%     
% Author: ginobuzz
%=========================================================================

    NUM_FILES = 10;
    DIR = 'features_train/'; % Directory path.
    EXT = '.txt';            % File extension.
    SPLIT_PERCENT = 0.6;     % Training:Validation = 0.6:0.4
    
    fprintf('> Starting Data Format...\n');
    tic;
    
    for i = 1:NUM_FILES
        
        % Load data.
        index = i - 1;
        file_path = strcat(DIR, num2str(index), EXT);
        try
            D = load(file_path);
        catch
            eMsg = strcat('> [ERROR] File not found (', file_path, ').');
            error(eMsg);
        end
        
        % Create matrix with labeled data.
        [M,N] = size(D);
        labelVec = ones(M,1) * index;
        L = horzcat(labelVec,D);
        
        % Split data into sets train/valid sets.
        split = floor(M * SPLIT_PERCENT);
        TrainSet{i} = L(1:split,:);
        ValidSet{i} = L((split + 1):M, :);
        
    end
    
    % Build output matracies.
    T = TrainSet{1};
    V = ValidSet{1};
    for i = 2:10
        T = vertcat(T,TrainSet{i});
        V = vertcat(V,ValidSet{i});
    end
    
    fprintf('> Data Format: Successful. [Operation took %f seconds]\n', toc);
end

