function [ yimp ] = impute_labels(pruning, tree, labelCounts, bestLabel)
% IMPUTE_LABELS(PRUNING, TREE, TREE_PARENTS, BESTLABEL) Impute
% labels for all leaves (points) based on current pruning and "best label"
% estimates.
%
% INPUT
%   pruning             Kx1 vector (list) of nodes in current pruning.
%   tree                cluster tree
%   labelCounts         per-node label counts
%   bestLabel           (2*M-1)x1 vector of "best" labels (one per node).
%
% OUTPUT
%   yimp             	Mx1 vector of imputed labels.
%
% Author: Dave Kale <dkale@usc.edu>
%         Anil Ramakrishna <akramakr@usc.edu>

yimp = zeros(tree.n_leaves,1);

if bestLabel(tree.root) == 0
    [~,bl] = max(labelCounts(tree.root,:));
    bestLabel(tree.root) = bl;
end

for i=1:length(pruning)
    v = pruning(i);
    leaves = get_leaves(tree, v);
    while bestLabel(v) == 0
        v = tree.parents(v);
    end
    yimp(leaves) = bestLabel(v);
end

assert(all(yimp > 0))

end
