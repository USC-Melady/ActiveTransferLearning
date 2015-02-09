function part = data_split_stratified(labels, frac)
% DATA_SPLIT_STRATIFIED(labels, frac) Randomly splits set of n elements into
% subsets as specified by frac, assuring each subset receives an equal
% number of each label.
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

uniq_lab = unique(labels);
part = zeros(size(labels));

for i=1:numel(uniq_lab)
    part(labels==uniq_lab(i)) = data_split(labels(labels==uniq_lab(i)), frac);
end

end
