function [ perf ] = run_passive_blitzer_atl(Xs, ys, ds, Xtr, ytr, ...
                                           budget, batch_size, base_learner, ...
                                           Xte, yte, alpha_t, alpha_s, params)
% RUN_PASSIVE_SIMPLE_ATL(XS, YS, DS, XTR, YTR, BUDGET, BATCH_SIZE,
% BASE_LEARNER, XTE, YTE, ALPHA_T, ALPHA_S, PARAMS) Convenience function to
% run an active transfer learning experiment using random sampling and a
% the transfer learning framework from Blitzer, et al., NIPS 2008 (i.e., a
% convex combination of empirical risks). This is actually related to the
% "simple" approach (pooling source and target data) but with instance
% weights that change as the number of queried target labels grows (in
% particular, the per-target point weight shrinks as more are queried). 
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
%   alpha_t         target domain weight
%   alpha_s         list of source domain weights
%   params          struct of optional parameters to control behavior of
%                   URS and base learner.
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

n_batches = ceil(budget/batch_size);
perf = struct();

h = base_learner(Xs, ys, ones(size(ys)), params);
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
    
    [wt,ws,dom_s] = compute_blitzer_tl_weights([ds; zeros(size(ytr(qix)))], alpha_t, alpha_s);
    w = zeros(size(ys));
    for di=1:length(dom_s)
        w(ds==dom_s(di)) = ws(di);
    end
    w = [ w; wt*ones(size(ytr(qix))) ];

    h = base_learner([Xs;Xtr(qix,:)], [ys;ytr(qix)], w, params);
    [yh,ydv] = h([Xs;Xtr(qix,:)]);
    perf(bi+1).train = compute_fave_metrics([ys;ytr(qix)],yh,ydv);
    [yh,ydv] = h(Xte);
    perf(bi+1).test = compute_fave_metrics(yte,yh,ydv);
    
    [tpr,~,ppv] = prc_stats_binormal(yte,ydv,true);
    perf(bi+1).prec_at_90 = ppv(max(find(tpr<0.9,1)-1,1));
end

end
