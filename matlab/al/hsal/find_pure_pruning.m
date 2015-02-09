function [ Q,D ] = find_pure_pruning(tree, y, eta)
% FIND_PURE_PRUNING(TREE, Y, ETA) Perform BFS search of cluster tree for a
% pruning with label purity no worse than ETA.
%
% INPUT
%   tree            cluster tree structure
%   y               full label vector for points in tree
%   eta             maximum label error for Q
%
% RETURNS
%   Q               pruning
%   D               list of depths for each node in Q
%
% AUTHOR:   David Kale (dkale@usc.edu)
% DATE:     2015-01-26

labels = sort(unique(y));
l = map_labels(y, labels);
lc = count_labels(tree, length(labels), l);

Q = tree.root;
D = 0;
wQ = min(lc(Q,:), [], 2);
eQ = sum(wQ) / sum(tree.leafCounts(Q));
while eQ > eta
    chg_best = -inf;
    pi_best = 0;
    for pi=1:length(Q)
        C = tree.children(Q(pi),:);
        if any(C~=0)
            wC = min(lc(C,:), [], 2);
            chg = wQ(pi)-sum(wC);
            if chg > chg_best
                chg_best = chg;
                pi_best = pi;
            end
        end
    end
    if pi_best == 0
        break
    end
    v = Q(pi_best);
    Q(pi_best) = [];
    Q = [ Q tree.children(v,:) ];
    wQ = min(lc(Q,:), [], 2);
    eQ = sum(wQ) / sum(tree.leafCounts(Q));
    d = D(pi_best);
    D(pi_best) = [];
    D = [ D d+1 d+1 ];
end

end

