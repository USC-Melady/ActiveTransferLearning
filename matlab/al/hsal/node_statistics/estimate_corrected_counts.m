function [ lb, ct, ub ] = estimate_corrected_counts(labelCounts, leafCounts)
% ESTIMATE_CORRECTED_COUNTS(LABELCOUNTS, LEAFCOUNT) Estimate "corrected"
% label count (out of the total number of leaves for node) plus confidence
% interval.
%
% INPUT
%   labelCounts         Lx1 vector of sampled label counts (L is number of
%                       unique labels).
%   leafCounts           count of total leaves (labeled or unlabeled).
%
% OUTPUT
%   lb                  lower bound on corrected count confidence interval.
%   lb                  estimated corrected count.
%   ub                  upper bound on corrected count confidence interval.
%
% Author: Dave Kale <dkale@usc.edu>
%         Anil Ramakrishna <akramakr@usc.edu>

sampleCount = sum(labelCounts, 2);

if sampleCount > 0
    frac = labelCounts / sampleCount;
    frac(isnan(frac)) = 0;
    corr = 1 - sampleCount / leafCounts;
    delta = corr / sampleCount + sqrt(corr * frac .* (1 - frac) / sampleCount);

    ct = frac * leafCounts;
    err = delta * leafCounts;
    lb = max(labelCounts, ct - err);
    ub = min(leafCounts - sampleCount + labelCounts, ct + err);
else
    lb = [ 0 0 ];
    ct = [ 0 0 ];
    ub = [ leafCounts leafCounts ];
end

end
