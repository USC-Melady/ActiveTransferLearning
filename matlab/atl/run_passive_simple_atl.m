function [ perf ] = run_passive_simple_atl(Xs, ys, ds, Xtr, ytr, ...
                                           budget, batch_size, base_learner, ...
                                           Xte, yte, params)
% RUN_PASSIVE_SIMPLE_ATL(XS, YS, DS, XTR, YTR, BUDGET, BATCH_SIZE,
% BASE_LEARNER, XTE, YTE, PARAMS) Convenience function to run an active
% transfer learning experiment using random sampling and a
% simple transfer learning heuristic: just pool all the data. This is
% surprisingly difficult to beat, even with sophisticated approaches.
% Trains on (Xs,ys) plus labeled or queried data from (Xtr,ytr) and
% computes performance using (Xte,yte).
%
% INPUT
%   Xs              M x P data design matrix of source data
%   ys              M x 1 vector of labels for Xs
%   ds              M x 1 vector of ints>0 indicating source domain of
%                   points in (Xs,ys).
%   Xtr             N x P data design matrix of training data
%   ytr             N x 1 vector of labels for Xtr.
%   budget          query budget
%   batch_size      number of queries to make for each batch
%   base_learner    learner from base-learners directory
%   Xte             N x P data design matrix of test data
%   yte             N x 1 vector of labels for Xte.
%   params          struct of optional parameters to control behavior of
%                   URS and base learner.
%       .ws         source instance weights
%       .wt         target instance weights
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
%
% AUTHOR:   David Kale (dkale@usc.edu)
% DATE:     2015-01-26

ws = ones(size(ys));
if isfield(params, 'ws')
    for di=1:size(params.wt,1)
        ws(ds==params.ws(di,1)) = params.ws(di,2);
    end
end

wt = ones(size(ytr));
if isfield(params, 'wt')
    wt(:) = params.wt;
end
                                       
n_batches = ceil(budget/batch_size);
perf = struct();

h = base_learner(Xs, ys, ws, params);
[yh,ydv] = h(Xs);
[yh2,ydv2] = h(Xte);
[tpr,~,ppv] = prc_stats_binormal(yte,ydv2,true);
perf(1).num_quer = 0;
perf(1).train = compute_fave_metrics(ys,yh,ydv);
perf(1).test  = compute_fave_metrics(yte,yh2,ydv2);
perf(1).prec_at_90 = ppv(find(tpr<0.9,1)-1);

for bi=1:n_batches
    end_ix = min(bi*batch_size, budget);
    perf(bi+1).num_quer = end_ix;
    
    qix = 1:end_ix;
    h = base_learner([Xs;Xtr(qix,:)], [ys;ytr(qix)], [ws;wt(qix)], params);
    [yh,ydv] = h([Xs;Xtr(qix,:)]);
    perf(bi+1).train = compute_fave_metrics([ys;ytr(qix)],yh,ydv);
    [yh,ydv] = h(Xte);
    perf(bi+1).test = compute_fave_metrics(yte,yh,ydv);
    
    [tpr,~,ppv] = prc_stats_binormal(yte,ydv,true);
    perf(bi+1).prec_at_90 = ppv(max(find(tpr<0.9,1)-1,1));
end

end

