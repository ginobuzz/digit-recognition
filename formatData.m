function [ X, T, L ] = formatData( dir )
%=========================================================================
% formatData: combines all 10 feature files into two matricies.
% 
%   Input: 
%       dir - name of the directory containing the input files.
%
%   Output:
%       X - Training Matrix of size NxD.
%       T - Target matrix of size NxM.
%       L - Label vector of size Nx1.
%     
%   Author: ginobuzz
%=========================================================================
    
    tic;

    X = [];% Training Matrix
    T = [];% Target Matrix
    L = [];% Label Vector

    for i = 1:10

        % Read file.
        index = i - 1;
        file_path = strcar(dir, num2str(index), '.txt');
        try
            F = load(file_path);
        catch
            eMsg = strcat('> [ERROR] File not found (', file_path, ').');
            error(eMsg);
        end

        % Insert column of ones to front, for bias.
        [n,d] = size(F);
        tmpX  = horzcat(ones(n,1), F);
        X     = vertcat(X,tmpX);

        % Build target boolean-matrix & append to T.
        [n,d] = size(tmpX);
        tmpT  = zeros(n,m);
        for j = 1:n
            tmpT(j,i) = 1;
        end
        T = vertcat(T,tmpT);

        % Build label vector & append to L.
        tmpL  = ones(n,1) * index;
        L     = vertcat(L,tmpL);

    end
    
    fprintf('> Data Format: Successful. [Operation took %f seconds]\n', toc);

end
