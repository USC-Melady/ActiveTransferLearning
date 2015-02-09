function [ tree ] = tree_from_parents(parents, varargin)

if nargin > 1
    weights = varargin{1};
else
    weights = [];
end

if min(parents) < 0
    parents = parents + 1;
end
assert(min(parents) == 0)

tree.n_clusters = (size(parents,1) - 1)/2;
tree.n_leaves   = tree.n_clusters + 1;
tree.n_nodes    = tree.n_leaves + tree.n_clusters;
tree.root       = tree.n_nodes;
tree.parents    = parents;
tree.children   = zeros(tree.n_nodes,2);
for v=1:size(parents,1)
    p = parents(v);
    if p ~= 0
        if tree.children(p,1) == 0
            tree.children(p,1) = v;
        else
            assert(tree.children(p,2)==0)
            tree.children(p,2) = v;
        end
    end
end

if isempty(weights)
    weights = ones(tree.n_leaves,1);
end
tree.leafCounts = [ weights ones(tree.n_leaves,1) ];
for v=1:(tree.n_nodes-1)
    tree.leafCounts(tree.parents(v)) = tree.leafCounts(tree.parents(v)) + tree.leafCounts(v);
end

end
