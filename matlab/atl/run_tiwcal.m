function [ perf, misc ] = run_tiwcal(Xs, ys, ds, Xtr, ytr, budget, ...
                                     batch_size, base_learner, ...
                                     Xte, yte, alpha_t, alpha_s, params)
% RUN_HATL(XS, YS, DS, XTR, YTR, BUDGET, BATCH_SIZE, BASE_LEARNER, XTE,
% YTE, PARAMS) Convenience function to run an active transfer learning
% experiment using transfer-accelereted, importance weighted consistent
% active learning, from our ICDM 2013 "Accelerating Active Learning with
% Transfer Learning" paper. Using (Xs,ys) and (Xtr,ytr) during the querying
% and training process, then computes performance using (Xte,yte). 
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
%   alpha_s         source domain weights
%   params          struct of optional parameters to control behavior of
%                   URS and base learner.
%    .ws            source instance weights
%    .wtr           target instance weights
%    .kernel        kernel function, in case you want to cluster and train
%                   classifiers in a kernel similarity space, rather than
%                   using Euclidean distance. Make sure it returns a N x N
%                   symmetric matrix.
%    .kparam        parameter to control kernel function (e.g., gamma for
%                   simple RBF)
%    .cluster_with_labels   whether to utilize source labels during
%                   clustering. This really seems to help. We do it with
%                   heuristic: train a classifier on (Xs,ys), then
%                   predict on Xtr. Then include ys and the decisions
%                   values for Xtr as an additional feature, which we
%                   weight heavily during clustering (so the cluster
%                   algorithm puts points with the same label together
%                   before considering other features).
%    .FLIMIT        a parameter to control the behavior of
%                   cluster_with_labels.
%
% RETURNS
%   perf            struct array containing performance info for each batch
%    .num_quer      number of queries made so far
%    .train         classification performance on labeled training data for
%                   classifier trained on all points with imputed labels
%                   (see output of "compute_fave_metrics" function)
%    .test          test set classification performance for classifier
%                   trained on all points with imputed labels 
%                   (see output of "compute_fave_metrics" function)
%    .prec_at_90    precision at 90% recall. Useful for class imbalanced
%                   medical data sets
%   misc            struct array containing IWCAL specific info, useful for
%                   understanding what it's doing and debugging.
%    .query         true or false: whether label was queried on iteration t
%    .Gbound        value of G's upper bound at iteration t
%    .Gt            value of G at iteration t
%    .Pt            probability of querying label on iteration t
%    .ws            importance weights for source points
%    .wt            importance weights for target points
%
% AUTHOR:   David Kale (dkale@usc.edu)
% DATE:     2015-01-26
                          
if isfield(params, 'c0')
    c0 = params.c0;
else
    c0 = 0.001;
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

params.m = length(ys);      % # source examples
params.alpha = alpha_t;         % alpha, controls source/target weighting

% append data, labels
Xst  = [ Xs; Xtr ];
yst = [ ys; ytr ];
                                  
[ yq, iw, alhist ] = iwcal_main(Xst, [ys; nan(size(ytr))], yst, ...
                                budget, base_learner, ...
                                @tiwcal_weights, ...
                                @tiwcal_gbound, ...
                                @tiwcal_query_probability, ...
                                c0, c1, c2, Xte, yte, params);

perf = struct();
perf(1).num_quer = 0;
h = base_learner(Xs, ys, ones(size(ys)), params);
[yh,ydv] = h(Xs);
perf(1).train = compute_fave_metrics(ys,yh,ydv);
[yh,ydv] = h(Xte);
perf(1).test  = compute_fave_metrics(yte,yh,ydv);
[tpr,~,ppv] = prc_stats_binormal(yte,ydv,true);
perf(1).prec_at_90 = ppv(find(tpr<0.9,1)-1);
for bi=1:n_batches
    perf(bi+1).num_quer = min(5*bi,size(ytr,1));
    t = perf(bi+1).num_quer + size(ys,1);
    qix = ~isnan(yq(1:t));
    assert(all(~isnan(yq(qix))))
    
    w = tiwcal_weights(iw(qix), t, params);
    
    h = base_learner(Xst(qix,:), yq(qix), w, params);
    [yh,ydv] = h(Xst(qix,:));
    perf(bi+1).train = compute_fave_metrics(yq(qix),yh,ydv);
    [yh,ydv] = h(Xte);
    perf(bi+1).test = compute_fave_metrics(yte,yh,ydv);
    [tpr,~,ppv] = prc_stats_binormal(yte,ydv,true);
    perf(bi+1).prec_at_90 = ppv(max(find(tpr<0.9,1)-1,1));
end

alhist = alhist((size(ys,1)+1):end,:);
misc = struct();
misc(1).query = false;
misc(1).Gbound = nan;
misc(1).Gt = nan;
misc(1).Pt = nan;
misc(1).ws = 1;
misc(1).wt = nan;
for qi=1:size(alhist,1)
    misc(qi+1).query = alhist(qi,1);
    misc(qi+1).Gbound = alhist(qi,2);
    misc(qi+1).Gt = alhist(qi,3);
    misc(qi+1).Pt = alhist(qi,4);
    
    t = qi + size(ys,1);
    qix = ~isnan(yq(1:t));
    assert(all(~isnan(yq(qix))))
    
    [ ~,ws,wt ] = tiwcal_weights(iw(qix), t, params);
    misc(qi+1).ws = ws; misc(qi+1).wt = wt;
end

end
