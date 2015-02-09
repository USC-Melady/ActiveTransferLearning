function e = iwerr(yhat, y, iw)
% IWERR(YHAT, Y, IW) Importance weighted empirical error of predictions
% yhat.
%
% INPUT
%   yhat        Nx1 vector of predicted labels
%   y           Nx1 vector of true labels
%   iw          Nx1 vector of importance weights
%
% RETURNS
%   e           importance weighted error
%
% MOST definitions of importance weighted error (e.g., the one used in IWAL
% CAL) are defined over ALL samples, including those with zero weights.
% Thus, you should not omit those (but rather ensure that they have zero
% weights in the iw vector). Otherwise, the "normalization" by length(y)
% will be wrong.
%
% Author: Dave Kale (dkale@usc.edu)

e = ((yhat ~= y)' * iw) / length(y);

end
