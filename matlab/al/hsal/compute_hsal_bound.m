function [errP, nquer] = compute_hsal_bound(errQ, nQ, dQ, eta, delta, ...
                                            batch_size, varargin)
% COMPUTE_HSAL_BOUND(ERRQ, NQ, DQ, ETA, DELTA, VARARGIN) Compute bound from
% Theorem 1 of NIPS 2008 "Hierarchical Sampling for Active Learning" paper
% by Dasgupta and Hsu.
%
% INPUT
%   errQ            true label error for Q
%   nQ              size of (number of clusters in) Q
%   dQ              depth of Q
%   eta             upper bound on Q's error
%   delta           probability of bound holding
%   batch_size      batch size
%   (optional)
%   n_classes       number of classes; default value is 2
%   bta             beta parameter from bound (controls admissibility);
%                   default value is 2 
%
% RETURNS
%   errP            upper bound on P's label imputation error
%   nquer           number of label queries required to find P
%
% AUTHOR:   David Kale (dkale@usc.edu)
% DATE:     2015-01-26

if nargin == 7
    n_classes = varargin{1};
    bta = 2;
else
    if nargin > 7
        n_classes = varargin{1};
        bta = varargin{2}
    else
        n_classes = 2;
        bta = 2;
    end
end

errP = (bta+1)*errQ + eta;
nquer = (bta+1)/(bta-1) * nQ/eta * log(2^dQ*n_classes*batch_size*nQ / (eta*delta));

end
