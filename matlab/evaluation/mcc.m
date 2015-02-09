function m = mcc(y, yhat, varargin)
% MCC(y, yhat) Calculates matthews correlation coefficient for binary
% classification tasks.
% 
% INPUT
%   y       		true labels
%   yhat    		predicted labels
%	varargin{1}		positive label; defaults to +1
% 
% RETURNS
%	m 				matthews correlation coefficient
% 
% MCC has a rather complicated formula involving true/false
% positives/negatives:
%
%                           TP * FN - FP * FN
%       M  = ---------------------------------------------------
%            sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
%
% AUTHOR:	Dave Kale (dkale@usc.edu), USC
% DATE:		2015-01-26

assert(nargin >= 2)
if nargin > 2
	pos_label = varargin{1};
else
	pos_label = 1;
end

tp = sum((y==pos_label) & (yhat==pos_label));
fp = sum(yhat==pos_label) - tp;
tn = sum((y~=pos_label) & (yhat~=pos_label));
fn = sum(yhat~=pos_label) - tn;

mcc_denom = sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn));
if mcc_denom == 0
    mcc_denom = 1;
end
m = (tp * tn - fp * fn) / mcc_denom;

end
