function [ fs, p, r ] = fscore(y, yhat, varargin)
% FSCORE(y, yhat, beta) Calculates precision, recall, and F-score for
% binary classification tasks.
% 
% INPUT
%   y       		true labels
%   yhat    		predicted labels
%	varargin{1}		beta parameter for F-score; defaults to 1 for F1
%	varargin{2}		positive label; defaults to +1
%
% RETURNS
%   fs      F-score
%   p       precision
%   r       recall
% 
% F-score is defined as
%
%            (1 + BETA^2) * PRECISION * RECALL
%       F  = ---------------------------------
%              (BETA^2 * PRECISION) + RECALL
%
% When beta = 1 (F1 or balanced F-score), this becomes the harmonic mean of
% precision and recall:
%
%            2 * PRECISION * RECALL
%       F1 = ----------------------
%              PRECISION + RECALL
%
% NOTE: assumes labels are numeric, where 1 is positive class and
% negative class is ~= 1.
%
% AUTHOR:	Dave Kale (dkale@usc.edu), USC
% DATE:		2015-01-26

assert(nargin >= 2)
switch nargin
	case 2
		beta = 1;
		pos_label = 1;
	case 3
    	beta = varargin{1};
    	pos_label = 1;
    otherwise
    	beta = varargin{1};
    	pos_label = varargin{2};
end

p = precision(y, yhat, pos_label);
r = recall(y, yhat, pos_label);

fs = (1 + beta^2) * (p * r) / ((beta^2 * p) + r);

end
