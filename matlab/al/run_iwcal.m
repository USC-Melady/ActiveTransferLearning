function [ perf, misc ] = run_iwcal(Xtr, ytr, budget, batch_size, ...
                                    base_learner, Xte, yte, params)
% RUN_IWCAL(XTR, YTR, BUDGET, BATCH_SIZE, BASE_LEARNER, XTE, YTE,
% PARAMS) Convenience function to run an active learning experiment using
% importance weighted consistent active learning (IWCAL), from the NIPS
% 2010 "Agnostic Active Learning Without Constraints" paper by Beygelzimer, 
% et al. Queries and trains using (Xtr,ytr) and computes performance using
% (Xte,yte).
%
% INPUT
%   Xtr             N x P data design matrix of training data
%   ytr             N x 1 vector of labels for Xtr.
%   budget          query budget
%   batch_size      number of queries to make for each batch
%   base_learner    learner from base-learners directory
%   Xte             N x P data design matrix of test data
%   yte             N x 1 vector of labels for Xte.
%   params          struct of optional parameters to control behavior of
%                   URS and base learner.
%    .w             instance weights
%    .{c0,c1,c2}    tuning parameters for IWCAL; see Daniel Hsu's
%                   dissertation for details on setting these.
%
% RETURNS
%   perf            struct array containing performance info for each batch
%    .num_quer      number of queries made so far
%    .train         classification performance on labeled training data
%                   (see output of "compute_fave_metrics" function)
%    .test          test set classification performance
%                   (see output of "compute_fave_metrics" function)
%    .prec_at_90    precision at 90% recall. Useful for class imbalanced
%                   medical data sets
%   misc            struct array containing IWCAL specific info, useful for
%                   understanding what it's doing and debugging.
%    .query         true or false: whether label was queried on iteration t
%    .Gbound        value of G's upper bound at iteration t
%    .Gt            value of G at iteration t
%    .Pt            probability of querying label on iteration t
%
% NOTE: IWCAL is an online, conservative active learning algorithm. It
% receives the points in a streaming fashion (and arbitrary order) and
% decides whether to query labels one at a time. In original concept, it
% does not use batches and so the batch_size argument is ignored. However,
% in principle one modify it to to perform its updates for each batch,
% rather than for each point.
%
% Also, neither this function nor iwcal_main shuffles the data (to generate
% a random ordering), so you'll need to do that outside.
%
% AUTHOR:   David Kale (dkale@usc.edu)
% DATE:     2015-01-26

if isfield(params, 'c0')
    c0 = params.c0;
else
    c0 = 128;
end
if isfield(params, 'c1')
    c1 = params.c1;
else
    c1 = 1;
end
if isfield(params, 'c2')
    c2 = params.c2;
else
    c2 = 1;
end
                               
n_batches = ceil(budget/batch_size);

[ yq, iw, alhist ] = iwcal_main(Xtr, nan(size(ytr)), ytr, budget, ...
                                base_learner, @simple_weights, ...
                                @iwcal_gbound, @iwcal_query_probability, ...
                                c0, c1, c2, [], [], params);

perf = struct();
perf(1).num_quer = 0;
perf(1).prec_at_90 = 0.5;
perf(1).train = generate_dummy_metrics(ytr);
perf(1).test  = generate_dummy_metrics(yte);
                            
for bi=1:n_batches
    perf(bi+1).num_quer = min(5*bi,size(ytr,1));
    qix = ~isnan(yq(1:perf(bi+1).num_quer));
    assert(all(~isnan(yq(qix))))
    
    h = base_learner(Xtr(qix,:), yq(qix), iw(qix), params);
    [yh,ydv] = h(Xtr(qix,:));
    perf(bi+1).train = compute_fave_metrics(yq(qix),yh,ydv);
    [yh,ydv] = h(Xte);
    perf(bi+1).test = compute_fave_metrics(yte,yh,ydv);
    [tpr,~,ppv] = prc_stats_binormal(yte,ydv,true);
    perf(bi+1).prec_at_90 = ppv(max(find(tpr<0.9,1)-1,1));
end

misc = struct();
misc(1).query = false;
misc(1).Gbound = nan;
misc(1).Gt = nan;
misc(1).Pt = nan;
for qi=1:size(alhist,1)
    misc(qi+1).query = alhist(qi,1);
    misc(qi+1).Gbound = alhist(qi,2);
    misc(qi+1).Gt = alhist(qi,3);
    misc(qi+1).Pt = alhist(qi,4);
end

end
