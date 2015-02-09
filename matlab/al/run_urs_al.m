function [ perf ] = run_urs_al(Xtr, ytr, budget, batch_size, ...
                               base_learner, Xte, yte, params)
% RUN_URS_AL(XTR, YTR, BUDGET, BATCH_SIZE, BASE_LEARNER, XTE, YTE, PARAMS)
% Convenience function to run an active learning experiment using
% uncertainty region sampling. Queries and trains using (Xtr,ytr) and
% computes performance using (Xte,yte).
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
%       .w          instance weights
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

if isfield(params, 'w')
    w = params.w;
else
    if isfield(params, 'wt')
        w = params.wt;
    else
        w = ones(size(ytr));
    end
end
                               
[ yq, quer_hist ] = urs_main(Xtr, nan(size(ytr)), ytr, budget, batch_size, ...
                            base_learner, params);

perf = struct();
queries = [];
for bi=1:numel(quer_hist)
    queries = [ queries; quer_hist{bi} ];
    perf(bi).num_quer = length(queries);
    
    yl = yq(queries);
    if isempty(yl) || length(unique(yl))<=1
        perf(bi).num_quer = 0;
        perf(bi).prec_at_90 = 0.5;
        perf(bi).train = generate_dummy_metrics(ytr);
        perf(bi).test  = generate_dummy_metrics(yte);
    else
        h = base_learner(Xtr(queries,:), ytr(queries), w(queries), params);
        [yh,ydv] = h(Xtr(queries,:));
        perf(bi).train = compute_fave_metrics(ytr(queries),yh,ydv);
        [yh,ydv] = h(Xte);
        perf(bi).test = compute_fave_metrics(yte,yh,ydv);
        [tpr,~,ppv] = prc_stats_binormal(yte,ydv,true);
        perf(bi).prec_at_90 = ppv(max(find(tpr<0.9,1)-1,1));
    end
end

end
