function [ auc, opt_fp, opt_tp ] = auroc(y, yscore, varargin)
% AUROC(y, yscore) Calculates area under the receiver operating
% characteristic (ROC) curve.
% 
% INPUT
%   y 				true labels
%   yscore  		decision values (probabilities, SVM output, etc.)
%	varargin{1}		positive label; defaults to +1
%
% RETURNS
%   auroc 			area under the ROC curve
%   opt_tp  		TP rate at the optimal point on the curve
%   opt_fp  		FP rate at the optimal point on the curve
% 
% The ROC curve plots true positive rate vs. false positive rate. When
% people say "AUC" they usually mean under the ROC curve. We use the
% built-in MATLAB command perfcurve to calculate AUC here.
%
% AUTHOR:	Dave Kale (dkale@usc.edu), USC
% DATE:		2015-01-26

assert(nargin >= 2)
if nargin > 2
	pos_label = varargin{1};
else
	pos_label = 1;
end

[x, y, t, auc, opt] = perfcurve(y, yscore, pos_label, 'xCrit', 'FPR', 'yCrit', 'TPR');
opt_fp = opt(1);
opt_tp = opt(2);

end
