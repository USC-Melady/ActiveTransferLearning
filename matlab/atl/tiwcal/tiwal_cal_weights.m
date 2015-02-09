function w = tiwal_cal_weights(iw, t, params)
% TIWAL_CAL_WEIGHTS(IW, T, PARAMS) Compute TIWAL CAL instance weights based
% on importance weights, # target points seen, # source points
% (params.m), and alpha (params.alpha)
%
% INPUT
%   iw              Nx1 vector of importance weights
%   t               number of (source and target) samples *seen* (not labeled)
%   params:         struct with additional named parameters
%     alpha         transfer parameter
%     m             number of source examples
%
% RETURNS
%   w               Nx1 vector of weights
%
% Compute weights for Transfer IWAL CAL.
%
% Author: Dave Kale (dkale@usc.edu)

alpha = params.alpha;
m     = params.m;
n     = t - m;
assert(n + m >= length(iw))

%% source weights
if m > 0 && alpha < 1
    ws = (n + m) / m * (1-alpha);
else
    ws = 0;
end

%% target weights
if n > 0 && alpha > 0
    wt = (n + m) / n * alpha;
else
    wt = 0;
end

%% combine with importance weights
w = iw .* [ ws * ones(m,1); wt * ones(length(iw)-m,1) ];

end
