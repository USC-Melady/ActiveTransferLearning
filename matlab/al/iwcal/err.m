function e = err(y, yhat)
% IWERR(YHAT, Y) Empirical error of predictions yhat.
%
% INPUT
%   yhat        Nx1 vector of predicted labels
%   y           Nx1 vector of true labels
%
% RETURNS
%   e           importance weighted error
%
% Author: Dave Kale (dkale@usc.edu)

e = sum(y~=yhat) / length(y);

end
