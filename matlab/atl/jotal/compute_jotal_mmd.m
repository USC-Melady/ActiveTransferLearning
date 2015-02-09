function [mmd] = compute_jotal_mmd(Xs, ws, Xl, Xu, kernel, varargin)

if numel(varargin)>0
    kparam = varargin{1};
else
    kparam = [];
end

if (isempty(Xs) && isempty(Xl)) || isempty(Xu)
    fprintf('WARNING: cannot compute MMD\n')
    mmd = NaN;
    return;
end

ns = size(Xs,1);
nu = size(Xu,1);
nl = size(Xl,1);

Kss = kernel(Xs,[],kparam);
Kuu = kernel(Xu,[],kparam);
Ksu = kernel(Xs,Xu,kparam);

if ~isempty(Xl)
    Kll = kernel(Xl,[],kparam);
    Ksl = kernel(Xs,Xl,kparam);
    Klu = kernel(Xl,Xu,kparam);
else
    Kll = zeros(1,1);
    Ksl = zeros(size(Xs,1),1);
    Klu = zeros(1,size(Xu,1));
end

kss = ws'*Kss*ws;
ksl = sum(ws'*Ksl);
kll = sum(Kll(:));
ksu = sum(ws'*Ksu);
klu = sum(Klu(:));
kuu = sum(Kuu(:));

mmd = (1/(ns+nl)^2) * (kss + 2*ksl + kll) ...
      - 2/((ns+nl)*nu) * (ksu + klu) ...
      + 1/nu^2 * kuu;

% w = [ ws; ones(size(Xl,1),1) ];
% Ksl_u = kernel([Xs; Xl],Xu,kparam);
% % [Ksl_u,sigma] = kernel([Xs; Xl],Xu);
% % Ksl_u = Ksl_u.^(1/sigma^2);
% Ksl_u = bsxfun(@times, w, Ksl_u);
% 
% Ksl = kernel([Xs;Xl],[],kparam);
% % Ksl = kernel([Xs;Xl],[]).^(1/sigma^2);
% Ksl = bsxfun(@times, w, bsxfun(@times, w', Ksl));
% ksl = squareform(Ksl-diag(diag(Ksl)), 'tovector');
% 
% Kuu = kernel(Xu,[],kparam);
% % Kuu = kernel(Xu,[]).^(1/sigma^2);
% kuu = squareform(Kuu-diag(diag(Kuu)), 'tovector');
% 
% mmd = mean(ksl) - 2 * mean(Ksl_u(:)) + mean(kuu);

end
