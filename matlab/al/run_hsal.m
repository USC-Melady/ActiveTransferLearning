function [ perf, misc ] = run_hsal(Xtr, ytr, budget, batch_size, ...
                                   base_learner, Xte, yte, params)
% RUN_HSAL(XTR, YTR, BUDGET, BATCH_SIZE, BASE_LEARNER, XTE, YTE,
% PARAMS) Convenience function to run an active learning experiment using
% hierarchical sampling for active learning (HSAL), from the ICML 2008
% "Hierarchical Sampling for Active Learning" paper by Dasgupta and Hsu.
% Queries and trains using (Xtr,ytr) and computes performance using
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
%    .kernel        kernel function, in case you want to cluster and train
%                   classifiers in a kernel similarity space, rather than
%                   using Euclidean distance. Make sure it returns a N x N
%                   symmetric matrix.
%    .kparam        parameter to control kernel function (e.g., gamma for
%                   simple RBF)
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
%   misc            struct array containing HSAL specific info, useful for
%                   understanding what it's doing and debugging.
%    .Q             an approximately "optimal" pruning
%    .dQ            Q's depth (depth of deepest node)
%    .purity_T      overall label purity of T, excluding root and leaves
%    .purity_Q      label purity of Q
%    .prunings      struct containing info about each batch's pruning P
%      .n_clusters  size of P
%      .purity      true underlying label purity of P
%      .err         actual label imputation error for P (based on only
%                   observed labels)
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
params.winit = w;

if isfield(params, 'kernel')
    kernel = params.kernel;
    if isfield(params, 'kparam')
        kparam = params.kparam;
    else
        kparam = [];
    end
else
    kernel = [];
end

if ~isempty(kernel)
    Dtr = squareform(1-kernel(Xtr,[],kparam), 'tovector');
    params.Z = linkage(Dtr, 'weighted');
else
    Dtr = pdist(Xtr);
    params.Z = linkage(Dtr, 'ward');
end

[ yq, quer_hist, prunings, yimp ] = hsal_main(Xtr, nan(size(ytr)), ytr, ...
                                              budget, batch_size, params);

T = tree_from_linkage(params.Z, w);
labels = sort(unique(ytr));
l = map_labels(ytr, labels);
lc = count_labels(T, length(labels), l);

if ~isfield(params, 'eta')
    params.eta = 0.05;
end

misc = struct();
[ Q, dQ ] = find_pure_pruning(T, l, params.eta);
misc.Q = Q;
misc.dQ = dQ;
misc.purity_Q = compute_label_purity(T,l,Q);
misc.purity_T = compute_label_purity(T,l);

perf = struct();
queries = [];
for bi=1:numel(quer_hist)
    misc.prunings(bi).n_clusters = length(prunings{bi});
    misc.prunings(bi).purity = sum(max(lc(prunings{bi},:), [], 2)) ...
                                        / sum(T.leafCounts(prunings{bi}));
    acc_etc = compute_fave_metrics(ytr, yimp(:,bi), yimp(:,bi));
    misc.prunings(bi).err = 1-acc_etc.a;
    
    queries = [ queries; quer_hist{bi} ];
    perf(bi).num_quer = length(queries);

    if ~isempty(base_learner) && ~isempty(Xte) && ~isempty(yte)
        yl = yq(queries);
        if length(unique(yimp(:,bi)))>1
            h = base_learner(Xtr, yimp(:,bi), w, params);
            if isempty(yl)
                perf(bi).train = generate_dummy_metrics(ytr);
            else
                [yh,ydv] = h(Xtr(queries,:));
                perf(bi).train = compute_fave_metrics(ytr(queries),yh,ydv);            
            end

            [yh,ydv] = h(Xte);
            perf(bi).test = compute_fave_metrics(yte,yh,ydv);
            [tpr,~,ppv] = prc_stats_binormal(yte,ydv,true);
            perf(bi).prec_at_90 = ppv(max(find(tpr<0.9,1)-1,1));
        else
            perf(bi).train = generate_dummy_metrics(ytr);
            perf(bi).test = generate_dummy_metrics(yte);
            perf(bi).prec_at_90 = 0.5;
        end

        if isempty(yl) || length(unique(yl))<=1
            perf(bi).train_2 = generate_dummy_metrics(ytr);
            perf(bi).test_2 = generate_dummy_metrics(yte);
            perf(bi).prec_at_90_2 = 0.5;
        else
            h = base_learner(Xtr(queries,:), yq(queries), w(queries), params);
            [yh,ydv] = h(Xtr(queries,:));
            perf(bi).train_2 = compute_fave_metrics(ytr(queries),yh,ydv);
            [yh,ydv] = h(Xte);
            perf(bi).test_2 = compute_fave_metrics(yte,yh,ydv);
            [tpr,~,ppv] = prc_stats_binormal(yte,ydv,true);
            perf(bi).prec_at_90_2 = ppv(max(find(tpr<0.9,1)-1,1));
        end
    end
end

end
