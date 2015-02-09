function [ perf, misc ] = run_hatl(Xs, ys, ds, Xtr, ytr, ...
                                   budget, batch_size, base_learner, ...
                                   Xte, yte, params)
% RUN_HATL(XS, YS, DS, XTR, YTR, BUDGET, BATCH_SIZE, BASE_LEARNER, XTE,
% YTE, PARAMS) Convenience function to run an active transfer learning
% experiment using hierarchical active transfer learning (HATL), from our
% SDM 2015 "Hierarchical Active Transfer Learning" paper, applies the HSAL
% algorithm (Dasgupta and Hsu, ICML 2008) to the problem of active transfer
% learning. Using (Xs,ys) and (Xtr,ytr) during the querying and training
% process, then computes performance using (Xte,yte). 
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

ws = ones(size(ys));
if isfield(params, 'ws')
    for di=1:size(params.wtr,1)
        ws(ds==params.ws(di,1)) = params.ws(di,2);
    end
end

wtr = ones(size(ytr));
if isfield(params, 'wtr')
    wtr(:) = params.wtr;
end

Xst = [Xs;Xtr];
yst = [ys;ytr];
wst = [ws;wtr];
params.winit = wst;

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

if isfield(params, 'cluster_with_labels')
    use_labels = params.cluster_with_labels;
else
    use_labels = false;
end

if use_labels
    h = base_learner(Xs, ys, ws, params);
    [~,ytd] = h(Xtr);
    if isfield(params, 'FLIMIT')
        FLIMIT = params.FLIMIT;
        ytd(ytd<-FLIMIT) = -FLIMIT; ytd(ytd>FLIMIT) = FLIMIT;
        lw = 2*FLIMIT * size(Xst,2);
    else
        lw = size(Xst,2);
    end
    if ~isempty(kernel)
        Dst = 1-kernel([ Xst lw*[ys;ytd] ], [], kparam);
        Dst = squareform(Dst - diag(diag(Dst)), 'tovector');
    else
        Dst = pdist([ Xst lw*[ys;ytd] ]);
    end
else
    if ~isempty(kernel)
        Dst = 1-kernel(Xst, [], kparam);
        Dst = squareform(Dst - diag(diag(Dst)), 'tovector');
    else
        Dst = pdist(Xst);
    end
end
if ~isempty(kernel)
    params.Z = linkage(Dst, 'weighted');
else
    params.Z = linkage(Dst, 'ward');
end

[ yq, quer_hist, prunings, yimp ] = hsal_main(Xst, [ys; nan(size(ytr))], ...
                                              yst, budget, batch_size, ...
                                              params);

T = tree_from_linkage(params.Z, wst);
Tt = tree_from_linkage(params.Z, [zeros(size(ws)); wtr]);
labels = sort(unique(yst));
l = map_labels(yst, labels);

if ~isfield(params, 'eta')
    params.eta = 0.05;
end

misc = struct();
[ Q, dQ ] = find_pure_pruning(T, l, params.eta);
misc.Q = Q;
misc.dQ = dQ;
misc.purity_Q = compute_label_purity(T,l,Q);
misc.purity_T = compute_label_purity(T,l);
misc.purity_Q_target = compute_label_purity(Tt,l,Q);
misc.purity_T_target = compute_label_purity(Tt,l);

perf = struct();
queries = [1:size(ys,1)]';
for bi=1:numel(quer_hist)
    misc.prunings(bi).n_clusters = length(prunings{bi});
    misc.prunings(bi).purity = compute_label_purity(T,l, prunings{bi});
    acc_etc = compute_fave_metrics(yst, yimp(:,bi), yimp(:,bi));
    misc.prunings(bi).err = 1-acc_etc.a;
    misc.prunings(bi).purity_target = compute_label_purity(Tt,l, prunings{bi});
    acc_etc = compute_fave_metrics(ytr, yimp((size(ys,1)+1):end,bi), ...
                                        yimp((size(ys,1)+1):end,bi));
    misc.prunings(bi).err_target = 1-acc_etc.a;
    
    queries = [ queries; quer_hist{bi} ];
    perf(bi).num_quer = length(queries)-size(ys,1);

    yl = yq(queries);
    if length(unique(yimp(:,bi)))>1
        h = base_learner(Xst, yimp(:,bi), wst, params);
        if isempty(yl)
            perf(bi).train = generate_dummy_metrics(yst);
        else
            [yh,ydv] = h(Xst(queries,:));
            perf(bi).train = compute_fave_metrics(yst(queries),yh,ydv);            
        end
        
        [yh,ydv] = h(Xte);
        perf(bi).test = compute_fave_metrics(yte,yh,ydv);
        [tpr,~,ppv] = prc_stats_binormal(yte,ydv,true);
        perf(bi).prec_at_90 = ppv(max(find(tpr<0.9,1)-1,1));
    else
        perf(bi).train = generate_dummy_metrics(yst);
        perf(bi).test = generate_dummy_metrics(yte);
        perf(bi).prec_at_90 = 0.5;
    end
    
    if isempty(yl) || length(unique(yl))<=1
        perf(bi).train_2 = generate_dummy_metrics(yst);
        perf(bi).test_2 = generate_dummy_metrics(yte);
        perf(bi).prec_at_90_2 = 0.5;
    else
        h = base_learner(Xst(queries,:), yq(queries), wst(queries), params);
        [yh,ydv] = h(Xst(queries,:));
        perf(bi).train_2 = compute_fave_metrics(yst(queries),yh,ydv);
        [yh,ydv] = h(Xte);
        perf(bi).test_2 = compute_fave_metrics(yte,yh,ydv);
        [tpr,~,ppv] = prc_stats_binormal(yte,ydv,true);
        perf(bi).prec_at_90_2 = ppv(max(find(tpr<0.9,1)-1,1));
    end
   
end

end
