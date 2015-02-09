function [ path ] = get_path_ascending(tree, descendent, ancestor)
% GET_PATH_ASCENDING(TREE, DESCENDENT, ANCESTOR) Retrieve node path from
% descendent to ancestor.
% 
% INPUT
%   tree            Cluster tree struct. Must have field named "parent.".
%                   The first is a (2M-1) vector representation of a
%                   hierarchical clustering of M points. The jth entry
%                   gives the children for the jth node in the tree. The
%                   latter is an M vector of node labels. 0 means label is
%                   unknown. The maximum allowed label is N_LABELS.
%                   Optionally, may have weights.
%   descendent          leaf node ID (1 <= leaf <= M).
%   ancestor            non-leaf node ID (m+1 <= node <= 2*M-1).
%
% OUTPUT
%   path                vector (list) of nodes from leaf to node,
%                       inclusive.
%
% Author: Dave Kale <dkale@usc.edu>
%         Anil Ramakrishna <akramakr@usc.edu>

assert(descendent > 0)
assert(descendent <= ancestor)
assert(ancestor <= tree.n_nodes)

path = descendent;
curr = descendent;
while curr ~= ancestor
    assert(curr ~= 0)
    curr = tree.parents(curr);
    path = [ path curr ];
end

end
