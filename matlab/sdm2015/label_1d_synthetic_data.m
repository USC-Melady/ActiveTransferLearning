function [y] = label_1d_synthetic_data(X)
% LABEL_1D_SYNTHETIC_DATA(N) Labels a 1D data set using the labeling
% function from the synthetic data experiments in our SDM 2015 paper,
% "Hierarchical Active Transfer Learning."
%
% INPUT
%   X          1D data set
%
% RETURNS
%   y          labels
% 
% AUTHOR:   Dave Kale (dkale@usc.edu), USC
% DATE:     2015-01-26

y = zeros(size(X,1),1);
y(X<=1.25) = -1;
y(X>1.25 & X<=2.75) = 1;
y(X>2.75 & X<=4.25) = -1;
y(X>4.25) = 1;

ix = rand(size(y))<0.01;
y(ix) = -y(ix);

end
