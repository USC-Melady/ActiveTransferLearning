function [ tree ] = tree_from_linkage(Z, varargin)

if nargin > 1
    weights = varargin{1};
else
    weights = [];
end

tree.n_clusters = size(Z,1);
tree.n_leaves   = tree.n_clusters + 1;
tree.n_nodes    = tree.n_leaves + tree.n_clusters;
tree.root       = tree.n_nodes;
tree.children   = [ zeros(tree.n_leaves,2); Z(:,1:2) ];
tree.parents    = zeros(tree.n_nodes, 1);
tree.depth      = zeros(tree.n_nodes, 1);
for v=1:tree.n_clusters
    tree.parents(Z(v,1)) = v + tree.n_leaves;
    tree.parents(Z(v,2)) = v + tree.n_leaves;
end

P = tree.root;
Dp = 0;
while ~isempty(P)
    v = P(1); P(1) = [];
    d = Dp(1); Dp(1) = [];
    if v > 0
        tree.depth(v) = d;
        P = [ P tree.children(v,:) ];
        Dp = [ Dp d+1 d+1 ];
    end
end

if isempty(weights)
    weights = ones(tree.n_leaves,1);
end
tree.leafCounts = [ weights; zeros(tree.n_clusters,1) ];
for v=1:(tree.n_nodes-1)
    tree.leafCounts(tree.parents(v)) = tree.leafCounts(tree.parents(v)) + tree.leafCounts(v);
end

end
