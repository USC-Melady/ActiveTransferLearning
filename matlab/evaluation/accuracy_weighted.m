function a = accuracy_weighted(y, yhat, w)
% ACCURACY(y, yhat) Calculates accuracy for binary classification task, with
%
% INPUT
%   y       true labels
%   yhat    predicted labels
%
% RETURNS
%	a 		accuracy
%
% Accuracy is the number of correct predictions (i.e., true positives
% plus true negatives) divded by all predictions
%
%              TP + TN
%   ACCURACY = -------
%                ALL
%
% NOTE: assumes labels are numeric.
%
% AUTHOR:	Dave Kale (dkale@usc.edu), USC
% DATE:		2015-01-26

a = sum(w'*(y==yhat)) / sum(w);

end
