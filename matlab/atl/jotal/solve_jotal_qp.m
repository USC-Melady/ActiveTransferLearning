function [bnew,queries,scores] = solve_jotal_qp(Xs, Xtl, Xtu, B, kernel, params)

if isfield(params, 'kparam')
    kparam = params.kparam;
else
    kparam = [];
end

if B >= size(Xtu,1)
    bnew = zeros(size(Xs,1),1);
    queries = 1:size(Xtu,1);
    scores = zeros(size(Xtu,1),1);
    return
end

nl = size(Xtl,1);
ns = size(Xs,1);
nu = size(Xtu,1);

c = (nl + ns + nu) / (nu - B);

Kss = kernel(Xs,[],kparam)  / c^2;
Kuu = kernel(Xtu,[],kparam);
Ksu = kernel(Xs,Xtu,kparam) / c;

kuu = (nl + ns + B) / (c^2 * (nu - B)) * sum(Kuu,2);
ksu = (nl + ns + B) / (c^2 * (nu - B)) * sum(Ksu,2);

if isempty(Xtl)
    ksl = zeros(ns,1);
    kul = zeros(nu,1);
else
    Ksl = kernel(Xs,Xtl,kparam); %exp(-pdist2(Xs,Xtl).^2);
    Kul = kernel(Xtu,Xtl,kparam); %exp(-pdist2(Xtu,Xtl).^2);
    ksl = sum(Ksl,2) / c^2;
    kul = sum(Kul,2) / c;
end

z = ones(nu,1);
% cvx_begin
cvx_begin quiet
    variables b(ns) a(nu)
    minimize( 0.5*[b;a]'*[Kss Ksu; Ksu' Kuu]*[b;a] + [ ksl-ksu; kul-kuu ]'*[b;a] )
    subject to
        0 <= b <= 1
        0 <= a <= 1
        a'*z == B
cvx_end

bnew = b;
[~,queries] = sort(a, 'descend');
queries = queries(1:B);
scores = a;

end
