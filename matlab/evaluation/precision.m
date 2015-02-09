function p = precision(y, yhat, varargin)
% PRECISION(y, yhat, varargin) Calculates precision for binary
% classification tasks.
%
% INPUT
%   y       		true labels
%   yhat    		predicted labels
%	varargin{1}		positive label; defaults to +1
% 
% RETURNS
%	p 				precision
%
% Precision, a.k.a. positive predictive value, is equal to the
% number of "true positives" divided by total number of predicted
% positives (i.e., true positives plus false positives):
%
%                  TP
%    PRECISION = -------
%                TP + FP
%
% AUTHOR:	Dave Kale (dkale@usc.edu), USC
% DATE:		2015-01-26

assert(nargin >= 2)
if nargin > 2
	pos_label = varargin{1};
else
	pos_label = 1;
end

p = sum((y==pos_label) & (yhat==pos_label)) / sum(yhat==pos_label);

end
