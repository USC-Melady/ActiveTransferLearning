function [ nodes ] = get_bfs_traversal(tree, node)
% GET_BFS_TRAVERSAL(TREE_CHILDREN, NODE) Get breadth-first traversal from a
% node to all of its leaves.
% 
% INPUT
%   tree       (M-1)x2 matrix representation of a hierarchical
%                       clustering of m points. The jth entry gives the
%                       children for the (j+M)th node in the tree. Same
%                       format as that returned by LINKAGE and most
%                       hierarchical clustering methods.
%   node                single node ID.
%
% OUTPUT
%   nodes               list of nodes in BFS traversal order from node to
%                       its leaves.
%
% Author: Dave Kale <dkale@usc.edu>
%         Anil Ramakrishna <akramakr@usc.edu>

nodes = [ ];
queue = node;
while ~isempty(queue)
    curr = queue(1);
    queue(1) = [];
    ch = tree.children(curr,:);
    assert(all(ch>0) || all(ch==0))
    if any(ch>0)
        queue = [ queue ch ];
    end
    nodes = [ nodes curr ];
end

end
