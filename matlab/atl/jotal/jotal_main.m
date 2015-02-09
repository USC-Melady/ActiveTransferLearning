function [ y, quer_hist, b_hist, w ] = jotal_main(X, y, oracle, budget, batch_size, params)
% HSAL_MAIN(X, TREE_CHILDREN, Y, ORACLE, BUDGET, BATCH, BASE_LEARNER,
%           XTEST, YTEST, PARAMS) Our implementation...
%
% INPUT
%   X                   NxD matrix of features for labeled (source) data.
%   y                   Nx1 vector of labels; NaN means unlabeled.
%   d                   Nx1 vector of ints representing domain; 0=target.
%   oracle              Nx1 vector of ALL labels, used as "oracle."
%   budget              label budget.
%   batch_size          batch size (make this many queries in each iteration).
%   params              parameters struct
%    .debug             whether to output debug data in "debug_info" struct
%    .verbosity         controls verbosity of print outs while running
%
% RETURNS
%   y               Mx1 vector of queried labels.
%   yimp            Mx1 vector of imputed labels.
%   pruning         vector (list) of nodes in final pruning.
%   alhist          (budget+1)x6 matrix of active learning history. The bth
%                   row contains state after b-1 queries (the first entry
%                   is before any queries have been made). The columns are
%                   (1) query leaf, (2) accuracy of imputed labels, (3)
%                   accuracy of per-node  "best label" estimates, (4)
%                   number of clusters in current pruning, (5) training set
%                   accuracy for classifier if base_learner provided, (6)
%                   test set accuracy for classifier if base_learner, test
%                   data provided.
%   debug_info      struct of debug info, if requested
%
% Author: Dave Kale <dkale@usc.edu>
%         Anil Ramakrishna <akramakr@usc.edu>

if isfield(params, 'dom')
    d = params.dom;
else
    d = zeros(size(y));
end

if isfield(params, 'kernel')
    kernel = params.kernel;
else
    kernel = @gaussian_kernel;
end

six = ~isnan(y) & (d>0);
Xs = X(six,:);
% ys = y(six);
bs = zeros(sum(six),1);

tix = (d==0);
tno = find(tix);
Xt = X(tix,:);
yt = y(tix);
ort = oracle(tix);

quer_hist = cell(1,1);
b_hist    = cell(1,1);
n_queries = 0;
batch_num = 1;

while n_queries < budget && any(isnan(yt))
    uix = isnan(yt);
    uno = find(uix);
    [bnew, queries, scores] = solve_jotal_qp(Xs, Xt(~uix,:), Xt(uix,:), batch_size, kernel, params);
    bs = bs + bnew;
    
    b_hist{batch_num+1} = bs;
    if ~isempty(queries)
        yt(uno(queries)) = ort(uno(queries));
        quer_hist{batch_num+1} = tno(uno(queries));
    else
        quer_hist{batch_num+1} = [];
    end
    
    n_queries = n_queries + length(queries);
    batch_num = batch_num + 1;
    if batch_size == 0
        break
    end
end

y(tix) = yt;
w = zeros(size(X,1),1);
w(tix) = 1;
w(six) = bs;

end
