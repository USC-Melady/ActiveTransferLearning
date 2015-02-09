function [ auc ] = auprc(y, yscore, varargin)
% AUPRC(y, yscore) Calculates area under the precision-recall curve.
% 
% INPUT
%   y       		true labels
%   yscore  		decision values (probabilities, SVM output, etc.)
%	varargin{1}		positive label; defaults to +1
%
% RETURNS
%   auroc   		area under the ROC curve
%   opt_tp  		TP rate at the optimal point on the curve
%   opt_fp  		FP rate at the optimal point on the curve
% 
% This is a slightly different AUC from the usual area under the ROC curve.
% Here we plot recall vs. precision.
%
% AUTHOR:	Dave Kale (dkale@usc.edu), USC
% DATE:		2015-01-26

assert(nargin >= 2)
if nargin > 2
	pos_label = varargin{1};
else
	pos_label = 1;
end

[x, y, t, auc] = perfcurve(y, yscore, pos_label, 'xCrit', 'reca', 'yCrit', 'prec');

end
