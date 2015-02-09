function [ y, iw, alhist ] = iwcal_main(X, y, oracle, budget, base_learner, ...
                                           weightsFn, gboundFn, queryProbFn, ...
                                           c0, c1, c2, Xte, yte, params)
% IWCAL_MAIN(X, Y, ORACLE, BUDGET, BASE_LEARNER, WEIGHTSFN, GBOUNDFN,
%               QUERYPROBFN, C0, C1, C2, XTE, YTE, PARAMS) Executes the
% Our implementation of the IWAL CAL algorithm from the paper "Agnostic
% Active Learning Without Constraints" by Beygelzimer, Hsu, Langford, and
% Zhang, NIPS 2010. It can also be used to run TIWAL CAL from our paper
% "Accelerating Active Learning with Transfer Learning" by Kale and Liu,
% ICDM 2013. Its behavior is controlled by the (weightsFn, gboundFn,
% queryProbFn) functions, the (c0, c1, c2) parameters, the base_learner
% function, and fields of params.
%
% INPUT
%   X               NxD matrix of features; training data
%   y               Nx1 vector of labels; NaN entries indicate unlabeled
%   oracle          Nx1 vector of labels, used as "oracle"
%   budget          label budget
%   baes_learner    base learner function from base-learners directory
%   weightsFn       function that takes at most three parameters and
%                   returns Nx1 vector of instance weights for training
%                   data. Valid candidates: simple_weights, tiwcal_weights
%   gboundFn        function that takes at most three parameters and
%                   returns Gbound_t (upper bound on disagreement for
%                   iteration t). Valid candidates: iwcal_gbound,
%                   tiwcal_gbound
%   queryProbFn     function that takes at most six parameters and returns
%                   P_t (query probability for iteration t). Valid
%                   candidates: iwcal_query_probability,
%                   tiwcal_query_probability
%   c0, c1, c2      IWAL CAL parameters
%   Xte             TxD matrix of features; test data; may be empty
%   yte             Tx1 vector of labels; test data; may be empty
%   params          parameters struct
%       .quiet      controls verbosity
%                   may contain other fields for TIWAL CAL
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
                                       
%% setup
% ensure that X, y, w, oracle all have same N
assert(size(X,1) == length(y))
assert(length(oracle) == length(y))

% check whether we have valid test data
assert(size(Xte,1) == length(yte))
do_test = ~isempty(Xte);

% check and sanitize parameters
assert(budget > 0)
assert(c0 > 0)
assert(c1 > 0)
assert(c2 > 0)

if ~isfield(params, 'quiet')
    params.quiet = 1;
end

% initialize importance weights
iw = zeros(size(y));
iw(~isnan(y)) = 1;

% start at zero queries, t = # labeled instances
qcount = 0;
t = find(isnan(y), 1, 'first');
assert(t-1 == sum(~isnan(y))) % ensure that labeled instances are at front

% storage of our active learning history
alhist = zeros(length(y), 5);
alhist(1:(t-1), :) = repmat([ 1 nan nan nan nan ], t-1, 1);

%% begin active learning loop
if params.quiet > 0
    fprintf('Begin active learning')
    if params.quiet > 1
        fprintf(':\n')
    end
end
sentinel = 0.01;
while t <= length(y) && qcount < budget
    idx = ~isnan(y(1:(t-1))); % indeces of labeled points so far
    
    % Might as well query until we have a label from each class
    if sum(idx) == 0 || length(unique(y(idx))) < 2
        do_query = 1;
        Gbound   = nan;
        G_t      = nan;
        P_t      = 1;
        err_t    = nan;
        err_tp   = nan;
        if do_test && t > 1
            alhist(t-1, 5) = 0.5;
        end
        
    else % we can do active learning now
        Xl = X(idx,:);                          % labeled points so far
        yl = y(idx);
        wl = weightsFn(iw(idx), t-1, params);   % get instance weights
        xc = X(t,:);                            % next unlabeled point
        
        % train our two hypotheses (ERM, alternative)
        [ h_t, h_tp ] = base_learner([ Xl; xc ], [ yl; nan ], [ wl; inf ], params);
        assert(h_t(xc) ~= h_tp(xc))    % shouldn't agree
        
        % compute test set error of h_t
        if do_test && t > 1
            [ yhat, ~ ] = h_t(Xte);
            alhist(t-1, 5) = err(yte, yhat);
        end
        
        % Gbound: upper bound on disagreement (i.e., deviation) in H
        Gbound = gboundFn(t, c0, params);

        % labels, weights for data seen so far (including unlabeled data)
        yt = zeros(t-1,1);
        yt(idx) = y(idx);
        wt = zeros(size(yt));                
        wt(idx) = wl;

        % NOTE on below: importance weighted error is computed over ALL
        % data points that we've seen so far, labeled or unlabeled. That's
        % why we made yt, wt above and put the predicted labels ypred into
        % larger vectors, e.g., yhat, below. The thing is that we don't
        % want to waste computation ACTUALLY predicting labels for
        % unlabeled data.

        % ERM predicted labels, importance weighted error
        yhat = zeros(size(yt));
        [ ypred, ~ ] = h_t(Xl);
        yhat(idx) = ypred;
        err_t  = iwerr(yhat, yt, wt);

        % alternative predicted labels, importance weighted error
        yhatp  = zeros(size(yt));
        [ ypred, ~ ] = h_tp(Xl);
        yhatp(idx) = ypred;
        err_tp = iwerr(yhatp, yt, wt);

        % G_t: estimate of disagreement (i.e., deviation) in H
        G_t = err_tp - err_t;
        
        % compare G_t to Gbound, compute query probability
        if abs(G_t) <= Gbound
            P_t = 1;
        else
            P_t = queryProbFn(t, abs(G_t), c0, c1, c2, params);
        end
        assert(P_t > 0 && P_t <= 1)
        
        % flip coin to decide whether to query label
        do_query = (rand(1) <= P_t);
    end
    
    if do_query > 0          % we should query
        y(t) = oracle(t);    % query label
        iw(t) = 1 / P_t;     % importance weight
        qcount = qcount + 1;
        
        if params.quiet > 1
            fprintf('IWAL: t=%4d err=%.3g erp=%.3g Gbd=%.3g G_t=%.3g P_t=%.3g qct=%4d y_t=%d iw=%.3g\n', ...
                    t, err_t, err_tp, Gbound, G_t, P_t, qcount, y(t), iw(t))
        end
    end
    
    % store coin toss outcome, Gbound, G_t, P_t in AL history
    alhist(t,1:4) = [ do_query Gbound G_t P_t ];
    t = t + 1;
    
    if params.quiet == 1
        if t / size(X,1) >= sentinel
            fprintf('.')
            sentinel = sentinel + 0.01;
        end
    end
end
if params.quiet > 0
    fprintf('\n')
end

% Compute and store test set error of final h_t
if do_test && t > 1
    idx = ~isnan(y);
    wl = weightsFn(iw(idx), t-1, params);
    [ h_t, ~ ] = base_learner(X(idx,:), y(idx), wl, params);
    [ yhat, ~ ] = h_t(Xte);
    alhist(t-1, 5) = err(yte, yhat);
end

end
