function G = gaussian_kernel(X1, X2, varargin)

assert(~isempty(X1) || ~isempty(X2))

if numel(varargin)>0
    gamma = varargin{1};
else
    gamma = [];
end

if ~isempty(X1) && ~isempty(X2)
    assert(size(X1,2)==size(X2,2))
    if isempty(gamma)
        gamma = 1/size(X1,1);
    end
    G = pdist2(X1,X2);
    G = exp(-gamma*G.^2);
else
    if ~isempty(X1)
        X = X1;
    else
        X = X2;
    end
    if isempty(gamma)
        gamma = 1/size(X,1);
    end
    G = squareform(pdist(X));
    G = exp(-gamma*G.^2);
end

end
