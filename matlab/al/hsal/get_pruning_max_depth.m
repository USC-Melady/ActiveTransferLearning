function dP = get_pruning_max_depth(tree, P)
% GET_PRUNING_MAX_DEPTH(TREE, P) Find depth of deepest node in P.
%
% INPUT
%   tree            cluster tree structure
%   P               pruning
%
% RETURNS
%   dP              depth of P
%
% AUTHOR:	David Kale (dkale@usc.edu)
% DATE:     2015-01-26

d = zeros(size(P));
for pi=1:length(P)
    path = get_path_ascending(tree, P(pi), tree.root);
    d(pi) = length(path)-1;
end

dP = max(d);

end
