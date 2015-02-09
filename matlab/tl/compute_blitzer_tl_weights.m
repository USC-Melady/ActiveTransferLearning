function [wt, ws, dom_s] = compute_blitzer_tl_weights(dom, alpha_t, alpha_s)
% COMPUTE_BLITZER_TL_WEIGHTS(DOM, ALPHA_T, ALPHA_S) Computes Blitzer-style
% transfer learning weights, as per the Blitzer, et al's NIPS 2008 paper.
% These are for performing transfer learning via convex combination of
% source and target risks.
%
% INPUT
%   dom         vector of ints indicating domain for each data point
%               (assuming all data are stacked into one array). Target
%               domain is indicated with a 0.
%   alpha_t     target risk weight
%   alpha_s     vector of source risk weights (order corresponds to
%               ascending order of source numbers)
%
% RETURNS
%   wt          target instance weight
%   ws          vector of source instance weights (order corresponds to
%               ascending order of source numbers)
%   dom_s       list of source numbers in ascending order
%
% AUTHOR: Dave Kale (dkale@usc.edu)
% DATE: 2015-01-26

assert(any(dom==0))
assert((alpha_t + sum(alpha_s)-1) < 1E-12)

if size(alpha_s,2) > size(alpha_s,1)
    alpha_s = alpha_s';
end

cts = tabulate(dom);
cts = cts(cts(:,2)>0,:);
dom_s = cts(2:end,1);
assert(length(dom_s) == length(alpha_s))

beta_t = cts(1,3)/100;
beta_s = cts(2:end,3)/100;
assert(length(beta_s) == length(alpha_s))

wt = alpha_t / beta_t;
ws = alpha_s ./ beta_s;

end
