function [ leaves ] = get_leaves(tree, node)
% GET_LEAVES(TREE_CHILDREN, NODE) Get leaves for a node.
% 
% INPUT
%   tree            Cluster tree struct. Must have field named "children"
%                   that is a (M-1)x2 matrix representation of a
%                   hierarchical clustering of M points. The jth entry
%                   gives the children for the (j+M)th node in the tree.
%                   Same format as that returned by LINKAGE and most
%                   hierarchical clustering methods.
%   node            single node ID.
%
% OUTPUT
%   parents         K vector (list) of parent nodes.
%
% Author: Dave Kale <dkale@usc.edu>
%         Anil Ramakrishna <akramakr@usc.edu>

assert(length(node) == 1)
assert(node > 0 && node <= tree.n_nodes)

leaves = get_bfs_traversal(tree, node);
leaves = sort(leaves(leaves<=tree.n_leaves));

end
