function [ gm, p, r ] = gmean_pr(y, yhat, varargin)
% GMEAN_PR(y, yhat) Calculates precision, recall, and geometric mean
% of the two for binary classification tasks.
% 
% INPUT
%   y       		true labels
%   yhat    		predicted labels
%	varargin{1}		positive label; defaults to +1
% 
% RETURNS
%   gm      geometric mean of precision, recall
%   p       precision
%   r       recall
% 
% The geometric mean of two numbers is equal to the square root of
% their product:
%            
%       GM = sqrt(PRECISION * RECALL)
%
% AUTHOR:	Dave Kale (dkale@usc.edu), USC
% DATE:		2015-01-26

assert(nargin >= 2)
if nargin > 2
	pos_label = varargin{1};
else
	pos_label = 1;
end

p = precision(y, yhat, pos_label);
r = recall(y, yhat, pos_label);

gm = sqrt(p * r);

end
