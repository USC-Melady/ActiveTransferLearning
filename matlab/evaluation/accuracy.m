function a = accuracy(y, yhat, varargin)
% ACCURACY(y, yhat) Calculates accuracy for binary classification tasks.
%
% INPUT
%   y       		true labels
%   yhat    		predicted labels
%	varargin{1}		instance weights
%
% RETURNS
%	a 				accuracy
%
% Accuracy is the number of correct predictions (i.e., true positives
% plus true negatives) divded by all predictions
%
%              TP + TN
%   ACCURACY = -------
%                ALL
%
% If weights are provided, then it becomes
%
%              SUM_i w_i * (y_i==yhat_i)
%   ACCURACY = -------------------------
%                	  SUM_i w_i
%
% AUTHOR:	Dave Kale (dkale@usc.edu), USC
% DATE:		2015-01-26

assert(nargin >= 2)
if nargin > 2
	w = varargin{1};
	a = (w' * (y==yhat)) / sum(w);
else
	a = sum(y==yhat) / length(y);
end

end
