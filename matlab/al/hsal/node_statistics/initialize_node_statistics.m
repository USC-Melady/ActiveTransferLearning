function [ stats ] = initialize_node_statistics(tree, n_labels, labels)

stats.labelCounts  = zeros(tree.n_nodes, n_labels);
stats.bestLabel    = zeros(tree.n_nodes);
stats.lb           = zeros(tree.n_nodes, n_labels);
stats.est          = zeros(tree.n_nodes, n_labels);
stats.ub           = repmat(tree.leafCounts, 1, 2);
stats.score        = tree.leafCounts;
stats.admissible   = zeros(tree.n_nodes, n_labels);
stats.canSplitNode = zeros(tree.n_nodes);

stats = bottom_up_statistics(tree, stats, labels);

end
