function G = cosine_kernel(X1, X2, varargin)

assert(~isempty(X1) || ~isempty(X2))
if ~isempty(X1) && ~isempty(X2)
    X1 = [ X1 ones(size(X1,1),1) ];
    X2 = [ X2 ones(size(X2,1),1) ];
    assert(size(X1,2)==size(X2,2))
    G = X1*X2';
    d1 = sqrt(sum(X1.^2,2));
    d2 = sqrt(sum(X2.^2,2));
    G = bsxfun(@rdivide, bsxfun(@rdivide, G, d1), d2');
else
    if ~isempty(X1)
        X = X1;
    else
        X = X2;
    end
    X = [ X ones(size(X,1),1) ];
    G = X*X';
    d = sqrt(sum(X.^2,2));
    G = bsxfun(@rdivide, bsxfun(@rdivide, G, d), d');
end

end
