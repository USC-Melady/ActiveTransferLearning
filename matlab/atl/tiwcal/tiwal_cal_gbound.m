function gbound = tiwal_cal_gbound(t, c0, params)
% TIWAL_CAL_GBOUND(T, C0, PARAMS) Compute the upper bound on disagreement
% at iteration t.
%
% INPUT
%   t               iteration of active learning
%   c0              constant
%   params:         struct with additional named parameters
%     alpha         transfer parameter
%     m             number of source examples
%
% Note that we drop several log and constant terms from the formula in the
% paper.
%
% Author: Dave Kale (dkale@usc.edu)

alpha = params.alpha;
m     = params.m;
t     = t - m;

if t > 1
    epst = c0 ./ (t-1); %c0 * log(t) / (t-1); % Daniel Hsu says drop log(t)
else
    epst = c0;
end
epsm = c0 / (2*m); %epsm = c0 * log(2) / (2*m); % we drop the log term, too

gbound = alpha * (sqrt(epst) + epst) + (1-alpha) * sqrt(epsm);

end
