function [ stats ] = bottom_up_statistics(tree, stats, labels)

n_labels = size(stats.labelCounts,2);
stats.labelCounts = count_labels(tree, n_labels, labels);

for v=1:tree.n_nodes
    [ lb, est, ub ]  = estimate_corrected_counts(stats.labelCounts(v,:), tree.leafCounts(v));
    stats.lb(v,:)  = lb;
    stats.est(v,:) = est;
    stats.ub(v,:)  = ub;

    LB = repmat(lb, n_labels, 1);
    UB = (repmat(2 * ub - tree.leafCounts(v), n_labels, 1) .* (1 - eye(n_labels)))';
    adm = all(LB > UB, 1);
    stats.admissible(v,:) = adm;

    if any(stats.admissible(v,:))
        [~, best] = max(stats.labelCounts(v,:) .* stats.admissible(v,:));
        stats.bestLabel(v) = best;
    else
        stats.bestLabel(v) = 0;
    end

    sc = tree.leafCounts(v);
    if stats.bestLabel(v) ~= 0
        % QUESTION: should score use ESTIMATE or LOWER BOUND?
%         sc = sc - est; % use estimate, a la paper
        sc = sc - lb(stats.bestLabel(v)); % use lower bound, a la DH's code
    end

    if v > tree.n_leaves
        ch = tree.children(v,:);
        sc_ch = stats.score(ch);
        sc_split = sum(sc_ch); % REGULARIZATION + 1;
        if sc_split < sc
            stats.score(v) = sc_split;
            stats.canSplitNode(v) = 1;
        else
            stats.score(v) = sc;
            stats.canSplitNode(v) = 0;
        end
    else
        stats.score(v) = sc;
        stats.canSplitNode(v) = 0;
    end
end

end
