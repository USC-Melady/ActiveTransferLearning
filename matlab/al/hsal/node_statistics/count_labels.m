function [ counts ] = count_labels(tree, n_labels, labels)
% COUNT_LABELS(TREE, N_LABELS) Get label counts for a cluster tree.
% 
% INPUT
%   tree            Cluster tree struct. Must have fields named "parent"
%                   and labels. The first is a (2M-1) vector representation
%                   of a hierarchical clustering of M points. The jth entry
%                   gives the children for the jth node in the tree. The
%                   latter is an M vector of node labels. 0 means label is
%                   unknown. The maximum allowed label is N_LABELS.
%                   Optionally, may have weights.
%
% OUTPUT
%   counts          (2M-1)x2xN_LABELS vector of label counts. For the second
%                   dimension, the first entry is weighted counts, and the
%                   second is raw counts. When tree has no weights, these
%                   are equal.
%
% By label count, we mean the number of labeled leaf nodes (points) that are
% "covered" by a node in our hierarchical clustering. The weighted count is
% what it sounds like.
%
% Author: Dave Kale <dkale@usc.edu>
%         Anil Ramakrishna <akramakr@usc.edu>

assert(all(labels >= 0 & labels <= n_labels))

counts = zeros(tree.n_nodes, n_labels);
for v=1:tree.n_leaves
    if labels(v) > 0
        counts(v,labels(v)) = tree.leafCounts(v);
    end
end

for v=1:(tree.n_nodes-1)
    counts(tree.parents(v),:) = counts(tree.parents(v),:) + counts(v,:);
end
% counts = counts((m+1):(m+n),:,:);

end
