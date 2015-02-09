function part = data_kfold(labels, k, stratified)
% DATA_KFOLD(labels, k, stratified) Wrapper for DATA_SPLIT: puts 1/k of
% instances into each of k sets (or "folds").
% 
% INPUT
%   labels      labels of instances
%   k           number of folds
%   stratified  should use stratification (for unbalanced data sets)
%
% RETURNS
%   part    vector length n listing to which subset each element belongs
% 
% For k-fold cross validation.
%
% AUTHOR:	Dave Kale (dkale@usc.edu), USC
% DATE:		2015-01-26

frac = repmat(1/k, 1, k);
if stratified == 1
    part = data_split_stratified(labels, frac);
else
    part = data_split(labels, frac);
end

end
