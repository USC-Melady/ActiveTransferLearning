function [ perf ] = run_passive_al(Xtr, ytr, budget, batch_size, ...
                                   base_learner, Xte, yte, params)
% RUN_PASSIVE_AL(XTR, YTR, BUDGET, BATCH_SIZE, BASE_LEARNER, XTE, YTE,
% PARAMS) Convenience function to run an active learning experiment using
% passive learning (i.e., random sampling). Queries and trains using
% (Xtr,ytr) and computes performance using (Xte,yte).
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
%   perf            struct array contain performance info for each batch
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
    if isfield(params, 'w')
        w = params.wt;
    else
        if isfield(params, 'winit')
            w = params.winit;
        else
            w = ones(size(ytr));
        end
    end
end
                               
n_batches = ceil(budget/batch_size);
perf = struct();

% quer_order = randperm(size(Xtr,1));
% yshuff = ytr(quer_order);
% qi = find(yshuff~=yshuff(1),1); q = quer_order(qi);
% quer_order(qi) = []; quer_order = [ q quer_order ];
% assert(ytr(quer_order(1)) ~= ytr(quer_order(2)))
% clear yshuff;

perf(1).num_quer = 0;
perf(1).prec_at_90 = 0.5;
perf(1).train = generate_dummy_metrics(ytr);
perf(1).test  = generate_dummy_metrics(yte);

for bi=1:n_batches
    end_ix = min(bi*batch_size, budget);
    perf(bi+1).num_quer = end_ix;
    
    qix = 1:end_ix;
    h = base_learner(Xtr(qix,:), ytr(qix), w(qix), params);
    [yh,ydv] = h(Xtr(qix,:));
    perf(bi+1).train = compute_fave_metrics(ytr(qix),yh,ydv);
    [yh,ydv] = h(Xte);
    perf(bi+1).test = compute_fave_metrics(yte,yh,ydv);
    [tpr,~,ppv] = prc_stats_binormal(yte,ydv,true);
    perf(bi+1).prec_at_90 = ppv(max(find(tpr<0.9,1)-1,1));
end

end
