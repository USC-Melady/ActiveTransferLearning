function [ yt, iwt, alhist ] = tiwal_cal(X, y, oracle, budget, base_learner, ...
                                         c0, c1, c2, Xte, yte, Xs, ys, alp, ...
                                         params)
% IWAL_CAL(X, Y, ORACLE, BUDGET, BASE_LEARNER, C0, C1, C2, XTE, YTE,
%          XS, YS, ALP, PARAMS) 
% Wrapper function that runs IWAL_CAL_MAIN with the following:
%
% * weights function:           @simple_weights
% * Gbound function:            @iwal_cal_gbound
% * query probability function: @iwal_cal_query_probability
%
% Together this achieves normal IWAL CAL behavior (vs. TIWAL CAL).
%
% INPUT
%   X               NxD matrix of features; training data
%   y               Nx1 vector of labels; NaN entries indicate unlabeled
%   oracle          Nx1 vector of labels, used as "oracle"
%   budget          label budget
%   baes_learner    base learner function from base-learners directory
%   c0, c1, c2      IWAL CAL parameters
%   Xte             TxD matrix of features; test data; may be empty
%   yte             Tx1 vector of labels; test data; may be empty
%   Xs              MxD matrix of features; source data
%   ys              Mx1 vector of labels; source labels
%   alp             transfer parameter, controls source/target weighting
%   params          parameters struct
%       .quiet      controls verbosity of IWAL_CAL_MAIN
%
% RETURNS
%   y               Nx1 vector of queried labels
%   iw              Nx1 vector of importance weights
%   alhist          Nx5 matrix of active learning history. The (t)th row
%                   contains: (1) 1/0 whether (t)th label was queried; (2)
%                   Gbound_t, (3) G_t, (4) P_t, (5) test set error, if Xte,
%                   yte provided
%
% Author: Dave Kale (dkale@usc.edu)

% check transfer learning setup
assert(size(Xs,1) == length(ys))
params.m = length(ys);      % # source examples
params.alpha = alp;         % alpha, controls source/target weighting

% append data, labels
X = [ Xs; X ];
y = [ ys; y ];
oracle = [ ys; oracle ];
                                  
[ y, iw, alhist ] = iwal_cal_main(X, y, oracle, budget, base_learner, ...
                                  @tiwal_cal_weights, ...
                                  @tiwal_cal_gbound, ...
                                  @tiwal_cal_query_probability, ...
                                  c0, c1, c2, Xte, yte, params);

% remove source weights, labels from y, iw
st = length(ys)+1;
yt = y(st:end);
iwt = iw(st:end);

% remove all but one source entries from alhist history
alhist = alhist((st-1):end,:);

end
