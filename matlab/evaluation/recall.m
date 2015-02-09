function r = recall(y, yhat, varargin)
% RECALL(y, yhat, varargin) Calculates recall for binary
% classification tasks.
%
% INPUT
%   y       		true labels
%   yhat    		predicted labels
%	varargin{1}		positive label; defaults to +1
% 
% RETURNS
%	r 				recall
%
% Recall, a.k.a. sensitivity or true positive rate, is equal to the
% number of "true positives" divided by total number of positives
% (i.e., true positives plus false negatives):
%
%                  TP
%       RECALL = -------
%                TP + FN
%
% AUTHOR:	Dave Kale (dkale@usc.edu), USC
% DATE:		2015-01-26

assert(nargin >= 2)
if nargin > 2
	pos_label = varargin{1};
else
	pos_label = 1;
end

r = sum((y==pos_label) & (yhat==pos_label)) / sum(y==pos_label);

end
