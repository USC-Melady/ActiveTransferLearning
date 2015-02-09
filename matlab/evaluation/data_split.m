function part = data_split(labels, frac)
% DATA_SPLIT(labels, frac) Randomly split instances into subsets as specified by
% frac.
% 
% INPUT
%   labels  labels of instances
%   frac    list: fraction of elements in each subset
%
% RETURNS
%   part    vector length n listing to which subset each element belongs
% 
% Helper function for splitting data sets for CV, etc.
%
% AUTHOR:	Dave Kale (dkale@usc.edu), USC
% DATE:		2015-01-26

assert(abs(sum(frac) - 1.0) < eps) % frac should sum to 1!

n = length(labels);

grp = [];
for fi=1:(length(frac)-1)
    grp = [ grp; repmat(fi, max(1, floor(frac(fi) * n)), 1) ];
end
grp = [ grp; repmat(length(frac), n - length(grp), 1) ];

inds = randperm(n);

[~, idx] = sort(inds);
part = grp(idx);

end
