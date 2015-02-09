function purity = compute_label_purity(tree, y, varargin)
% COMPUTE_LABEL_PURITY(TREE, Y, VARARGIN) Compute true label purity for a
% list of nodes or for a full tree, excluding root and leaves.
%
% INPUT
%   tree            cluster tree structure
%   y               full vector of labels
%   (optional)
%   P               list of nodes; otherwise, use full tree excluding root
%                   and leaves
%
% RETURNS
%   purity          label purity
%
% AUTHOR:   David Kale (dkale@usc.edu)
% DATE:     2015-01-26

if nargin > 2
    P = varargin{1};
else
    P = find((tree.children(:,1)>0)&(tree.parents>0));
end

labels = sort(unique(y));
l = map_labels(y, labels);
lc = count_labels(tree, length(labels), l);

purity = sum(max(lc(P,:), [], 2)) / sum(tree.leafCounts(P));

end
