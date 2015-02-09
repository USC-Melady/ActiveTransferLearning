function prob_t = tiwal_cal_query_probability(t, G_t, c0, c1, c2, params)
% TIWAL_CAL_QUERY_PROBABILITY(T, G_T, C0, C1, C2, PARAMS) Compute query
% probability at iteration t.
%
% INPUT
%   t               iteration of active learning
%   G_t             disagreement between ERM, alternative hypotheses
%   c0, c1, c2      IWAL CAL constants
%   params:         struct with additional named parameters
%     alpha         transfer parameter
%     m             number of source examples
%
% RETURNS
%   prob            query probability, 0 < prob <= 1
%
% Note that we drop several log and constant terms from the formula in the
% paper.
%
% Author: Dave Kale (dkale@usc.edu)

alpha = params.alpha;
m     = params.m;
t     = t - m;

if t > 0
    epst = c0 / (t-1); %c0 * log(t) / (t-1); % Daniel Hsu says drop log(t)
else
    epst = c0;
end
epsm = c0 / (2*m); %c0 * log(2) / (2*m); % we too drop the log(2) term

a = G_t + alpha * ((c1-1)*sqrt(epst) + (c2-1)*epst) + (alpha-1) * sqrt(epsm);
b = -alpha * c1 * sqrt(epst);
c = -alpha * c2 * epst;

% ye old quadratic formula
s1 = (-b + sqrt(b^2 - 4*a*c)) / (2*a);
s2 = (-b - sqrt(b^2 - 4*a*c)) / (2*a);

% we need at least one nonzero root...if we don't get one, then something
% has gone horribly wrong
if ~((s1 > 0 && s1 <= 1) || (s2 > 0 && s2 <= 1))
    fprintf('m\tt\tc0\tG_t\ts1\ts1^2\ts2\ts2^2\n')
    fprintf('%4d\t%4d\t%+.3g\t%+.3g\t%+.3g\t%+.3g\t%+.3g\t%+.3g\n', [ m t c0 G_t s1 s1^2 s2 s2^2 ])
end
assert((s1 > 0 && s1 <= 1) || (s2 > 0 && s2 <= 1))

if s1 >= 0
    prob_t = s1^2;
else
    prob_t = s2^2;
end

end
