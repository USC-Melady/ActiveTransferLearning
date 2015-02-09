function [ dist_hl, dist_err ] = approx_da_distance(Xs, Xt, learner, varargin)
% APPROX_DA_DISTANCE(Xs, Xt, varargin) Calculates an approximation to "dA"
% distance, as in Blitzer, et al's NIPS 2008 paper. Otherwise known as the
% "domain separator hypothesis."
%
% INPUT
%   Xs          N x D matrix of source features
%   Xt          N x D matrix of target features
%   learner     base learner to use as domain separator hypothesis.
%   (optional)
%   params      struct containing parameters to control learner's behavior
%   ws          instance weights for Xs
%   wt          instance weights for Xt
%
% RETURNS
%   dist_hl     hinge loss-based dA distance
%   dist_err    0-1 loss-based dA distance
% 
% This function calculates an approximation to the "dA" distance between two
% data domains, as in Blitzer, Crammer, Kulezsa, Pereira, and Wortman,
% "Learning Bounds for Domain Adaptation," NIPS 2008. Intuitively, dA
% distance quantifies the "separability" (as in the binary classificatin
% sense) of two domains. The more "separable" two data sets are, the
% greater the dA distance and the "less similar" they are. If it's tough to
% separate two data sets, then the lower the dA distance will be. dA
% distance depends on the hypothesis class, but it can be computed with
% only unlabeled samples from the domains, which is convenient.
%
% Blitzer, et al., approximate dA distance by "training a linear classifier
% to discriminate between the two domains. We use a standard hinge loss
% (normalized by the number of instances) and apply the quantity
% 1 - (hinge loss)." We use both hinge loss and 0-1 error. We can compute
% dA distance for arbitrary learners that use our "base learner" interface.
%
% AUTHOR: Dave Kale (dkale@usc.edu)
% DATE: 2015-01-26

switch numel(varargin)
    case 0
        params = struct();
        ws = ones(size(Xs,1),1);
        wt = ones(size(Xt,1),1);
    case 1
        params = varargin{1};
        ws = ones(size(Xs,1),1);
        wt = ones(size(Xt,1),1);
    case 2
        params = varargin{1};
        ws = varargin{2};
        wt = ones(size(Xt,1),1);
    otherwise
        params = varargin{1};
        ws = varargin{2};
        wt = varargin{3};
end
if isempty(ws)
    ws = ones(size(Xs,1),1);
end
if isempty(wt)
    wt = ones(size(Xt,1),1);
end

k = 10;
X = [Xs;Xt];
y = [ones(size(Xs,1),1); -ones(size(Xt,1),1)];
w = [ws; wt];
f = data_kfold(y, k, true);
err = zeros(1,k);
hloss = zeros(1,k);
for fi=1:k
    h  = learner(X(f~=fi,:), y(f~=fi), w(f~=fi), params);
    [ yh, ydv ] = h(X(f==fi));
    err(fi) = sum(yh~=y(f==fi)) / sum(f==fi);
    hloss(fi) = sum(min(1, max(0, 1 - y(f==fi).*ydv))) / sum(f==fi);
end

dist_err = 1 - mean(err);
dist_hl = 1 - mean(hloss);

end
