function [ pruning ] = prune_tree(pruning, to_prune, tree, canSplitNode)
% PRUNE_TREE(PRUNING, TO_PRUNE, TREE, CANSPLITNODE,
%            RECURSIVEPRUNE) Perform pruning step. For each selected node
%            (i.e., had a leaf label queried), decide whether to replace it
%            with descendents.
%
% INPUT
%   pruning             Kx1 vector (list) of nodes in current pruning.
%   to_prune            list of selected nodes (to consider for pruning).
%   tree                cluster tree using our special notation
%   canSplitNode        (2*M-1)x1 boolean vector, whether can split each
%                       node.
%
% OUTPUT
%   pruning             Kx1 vector (list) of nodes in updated pruning.
%
% Author: Dave Kale <dkale@usc.edu>
%         Anil Ramakrishna <akramakr@usc.edu>

while ~isempty(to_prune)
    v = to_prune(1);
    pruning = setdiff(pruning, v);
    to_prune(1) = [];
    if canSplitNode(v)
        % QUESTION: should we perform recursive pruning? I.e., if we
        % replace node with its children, should we prune them, too? Paper
        % doesn't discuss this, but Daniel's code does!
        to_prune = [ tree.children(v,:) to_prune ]; % yes, a la Daniel's code
%         pruning = [ get_children(tree_children, v) pruning ]; % no
    else
        pruning = [ v pruning ];
    end
end

pruning = sort(pruning);

end
