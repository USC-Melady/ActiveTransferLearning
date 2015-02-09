function prob = iwcal_query_probability(t, G_t, c0, c1, c2, ~)
% IWAL_CAL_QUERY_PROBABILITY(T, G_T, C0, C1, C2, PARAMS) Compute query
% probability at iteration t.
%
% INPUT
%   t               iteration of active learning
%   G_t             disagreement between ERM, alternative hypotheses
%   c0, c1, c2      IWAL CAL constants
%
% RETURNS
%   prob            query probability, 0 < prob <= 1
%
% Note that we drop several log and constant terms from the formula in the
% paper.
%
% Author: Dave Kale (dkale@usc.edu)

epst = c0 / (t-1); %c0 * log(t) / (t-1); % Daniel Hsu says drop the log(t)

a = G_t + (c1-1)*sqrt(epst) + (c2-1)*epst;
b = -c1 * sqrt(epst);
c = -c2 * epst;

% ye old quadratic formula
s1 = (-b + sqrt(b^2 - 4*a*c)) / (2*a);
s2 = (-b - sqrt(b^2 - 4*a*c)) / (2*a);

% we need at least one nonzero root...if we don't get one, then something
% has gone horribly wrong
if ~((s1 > 0 && s1 <= 1) || (s2 > 0 && s2 <= 1))
    fprintf('t\tc0\tG_t\ts1\ts1^2\ts2\ts2^2\n')
    fprintf('%4d\t%+.3g\t%+.3g\t%+.3g\t%+.3g\t%+.3g\t%+.3g\n', [ t c0 G_t s1 s1^2 s2 s2^2 ])
end
assert((s1 > 0 && s1 <= 1) || (s2 > 0 && s2 <= 1))

if s1 >= 0
    prob = s1^2;
else
    prob = s2^2;
end

end
