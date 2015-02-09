function [errPt,dAterm,sqterm] = compute_hatl_bound(errP, Ns, Nt, dA, ...
                                                    delta, varargin)
% COMPUTE_HATL_BOUND(ERRP, NS, NT, DA, DELTA, VARARGIN) Compute bound from
% Theorem 1 of our SDM 2015 "Hierarchical Active Transfer Learning" paper.
%
% INPUT
%   errP            upper bound on label error for pruning P, given by
%                   Theorem 1 of Dasgupta and Hsu, NIPS 2008.
%   Ns              Number of source points
%   Nt              Number of target points
%   dA              dA distance between source and target domains
%   delta           probability of bound holding
%   (optional)
%   ws              source instance weights
%   wt              target instance weights
%   errST           combined risk of ideal classifier; default value is 0.
%   vcdim           VC dimension of hypothesis space; default value is 1.
%
% RETURNS
%   errPt           upper bound on P's target-only label imputation error
%                   (very loose)
%   dAterm          value of dA distance term in bound; probably most
%                   useful quantity returned by this function.
%   sqterm          value of square root term in bound; usually quite large
%                   except for big data sets
%
% AUTHOR:   David Kale (dkale@usc.edu)
% DATE:     2015-01-26

if nargin > 5
    switch nargin
        case 6
            ws = varargin{6};
            wt = ones(Nt,1);
            errST = 0;
            vcdim = 1;
        case 7
            ws = varargin{6};
            wt = varargin{7};
            errST = 0;
            vcdim = 1;
        case 8
            ws = varargin{6};
            wt = varargin{7};
            errST = varargin{8};
            vcdim = 1;
        otherwise
            ws = varargin{6};
            wt = varargin{7};
            errST = varargin{8};
            vcdim = varargin{8};
    end
else
    ws = ones(Ns,1);
    wt = ones(Nt,1);
    errST = 0;
    vcdim = 1;
end

al = sum(wt)/(sum(ws) + sum(wt));

dAterm = (1-al) * (dA/2 + errST);
sqterm = 2 * sqrt((vcdim * log(2*Nt) - log(delta))/(2*Nt));
errPt = errP + dAterm + sqterm;

end
