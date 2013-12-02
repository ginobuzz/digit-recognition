function [ X, T ] = formatData( dir )
%=========================================================================
% formatData: combines all 10 feature files into three matricies.
% 
%   Input: 
%       dir - name of the directory containing the input files.
%
%   Output:
%       X - Training Matrix of size NxD.
%       T - Target matrix of size NxM.
%     
%   Author: ginobuzz
%=========================================================================
    
    tic;

    X = [];% Training Matrix
    T = [];% Target Matrix

    for i = 1:10

        % Read file.
        index = i - 1;
        file_path = strcat(dir, num2str(index), '.txt');
        try
            F = load(file_path);
        catch
            eMsg = strcat('> [ERROR] File not found (', file_path, ').');
            error(eMsg);
        end
        X = vertcat(X,F);

        % Build target boolean-matrix & append to T.
        [n,d] = size(F);
        tmpT  = zeros(n,10);
        for j = 1:n
            tmpT(j,i) = 1;
        end
        T = vertcat(T,tmpT);

    end
    
    fprintf('> Data Format: Successful. [Operation took %f seconds]\n', toc);

end
