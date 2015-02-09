function [ probVector ] = get_query_probabilities_active(pruning, tree, stats)
% QUERY_PROB_ÅCTIVE(PRUNING, TREE, STATS)
% Calculate probability of querying a leaf label from each node in our
% current pruning.
%
% INPUT
%   pruning             Kx1 vector (list) of nodes in current pruning.
%   tree                cluster tree structure
%   stats               cluster tree statistics structure
%
% OUTPUT
%   probVector          Kx1 vector of probabilities, one for each node in
%                       pruning (sums to 1).
%
% Implements the "active" strategy from the paper, where probability is
% proportional to product of node size and inversely proportional to
% estimated node label purity.
%
% Author: Dave Kale <dkale@usc.edu>
%         Anil Ramakrishna <akramakr@usc.edu>

probVector = tree.leafCounts(pruning,1) - stats.lb(pruning,1,1);

% ix = find(stats.bestLabel(pruning,1)~=0);
% for i=1:length(ix)
%     v = pruning(ix(i));
%     probVector(ix(i)) = probVector(ix(i)) - stats.lb(v,stats.bestLabel(v,1),1);
% end

ix = sum(stats.labelCounts(pruning,:),2)==tree.leafCounts(pruning);
probVector(ix) = 0;

probVector = probVector / sum(probVector);

end
