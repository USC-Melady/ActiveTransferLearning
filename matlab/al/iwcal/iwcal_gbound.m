function gbound = iwcal_gbound(t, c0, ~)
% IWAL_CAL_GBOUND(T, C0) Compute the upper bound on disagreement at
% iteration t.
%
% INPUT
%   t               iteration of active learning
%   c0              constant
%
% RETURNS
%   gbound          upper bound on disagreement at iteration t
%
% Note that we drop several log and constant terms from the formula in the
% original IWALCAL paper, as per the suggestion in Daniel Hsu's thesis.
%
% AUTHOR:	Dave Kale (dkale@usc.edu), USC
% DATE:		2015-01-26

epst = c0 ./ (t-1); %c0 * log(t) / (t-1);  % Daniel Hsu says drop the log(t)
gbound = sqrt(epst) + epst;

end
