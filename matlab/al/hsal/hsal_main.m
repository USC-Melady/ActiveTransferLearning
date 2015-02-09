function [ y, quer_hist, prunings, yimp ] = hsal_main(X, y, oracle, budget, batch_size, params)
% HSAL_MAIN(X, Y, ORACLE, BUDGET, BATCH_SIZE, PARAMS) Our implementation of HSAL.
%
% INPUT
%   X                   MxD matrix of features; training data.
%   y                   Mx1 vector of labels; NaN means unlabeled.
%   oracle              Mx1 vector of ALL labels, used as "oracle."
%   budget              label budget.
%   batch_size          batch size (make this many queries before pruning).
%   params              struct of optional parameters to control behavior
%    .debug             whether to output debug data in "debug_info" struct
%    .verbosity         controls verbosity of print outs while running
%    .Z                 hierarchical clustering of data
%
% RETURNS
%   y               Mx1 vector of queried labels.
%   quer_hist       cell array of queries for each batch.
%   pruning         cell array of prunings for each batch.
%   yimp            Mx1 vector of imputed labels.
%
% Author: Dave Kale <dkale@usc.edu>
%         Anil Ramakrishna <akramakr@usc.edu>
                                                        
%% SETUP

assert(size(X,1) == size(y,1))
assert(size(y,1) == size(oracle,1))
assert(budget > 0)
assert(batch_size > 0 && batch_size <= budget)

n_batches = ceil(budget / batch_size);
query_num = 0;
batch_num = 0;

% Level of verbosity:
%  0: prints nothing
%  1: small number of messages about where algorithm is
%  2: prints per-batch messages
%  3: prints per-query messages
if isfield(params, 'verbosity')
    verbosity = params.verbosity;
else
    verbosity = 0;
end

if isfield(params, 'Z')
    Z = params.Z;
else
    Z = linkage(pdist(X), 'ward');
end
assert(size(Z,1) == (size(X,1)-1))
assert(size(Z,2)>=2)

if isfield(params, 'winit')
    winit = params.winit;
else
    if isfield(params, 'w')
        winit = params.w;
    else
        winit = ones(size(y));
    end
end
assert(all(size(winit) == size(y)));

tree = tree_from_linkage(Z, winit);

% LABELS
%  oracle:      vector of all true labels
%  labels:      sorted list of labels (maps number to original label)
%  uniq_labels: number of unique labels
%  y:           provided or queried, 1 is pos class, nan is not labeled yet
%  yimp:        imputed labels
assert(sum(isnan(oracle))==0)
labels = sort(unique(oracle));
n_labels = length(labels);
assert(n_labels>1)
y = map_labels(y, labels);
oracle = map_labels(oracle, labels);

% Initialize node statistics
if verbosity > 2
    fprintf('Initialize node statistics with %d starting labels...', sum(y~=0))
    tic;
end
stats = initialize_node_statistics(tree, n_labels, y);
% if any(~isnan(y))
%     tree_src = tree_from_linkage(Z, 1*(y>0));
%     stats_src = initialize_node_statistics(tree_src, n_labels, y);
%     stats.bestLabel(:,1) = stats_src.bestLabel(:,1);
%     stats.canSplitNode(:,1) = stats_src.canSplitNode(:,1);
% end
if verbosity > 2
    el = toc;
    fprintf('%.2f sec\n', el)
end

quer_hist = cell(1,n_batches+1);
quer_hist(1) = [];

% Update pruning
if verbosity > 2
    fprintf('Performing initial pruning...')
end
pruning = prune_tree(tree.root, tree.root, tree, stats.canSplitNode);
if verbosity > 2
    fprintf('%d clusters:', length(pruning))
    for pi=1:length(pruning)
        fprintf(' %4d', pruning(pi))
    end
    fprintf('\n')
end
prunings = cell(1,n_batches+1);
prunings{1} = pruning;

% Impute labels
yimp      = zeros(size(y,1),n_batches+1);
yimp(:,1) = impute_labels(pruning, tree, stats.labelCounts, stats.bestLabel);

%% BEGIN ACTIVE LEARNING QUERY LOOP
if verbosity > 0
    fprintf('Begin active learning loop')
end
if verbosity > 1
    fprintf('\n')
end

while 1
    batch_num = batch_num + 1;
    if verbosity > 1
        fprintf('Batch %4d\n', batch_num)
    else
        if verbosity > 0
            fprintf('.')
        end
    end
    
    queries = [];
    selectedNodes = zeros(1, batch_size);
    for b=1:batch_size
        if query_num >= budget
            if verbosity > 0
                fprintf('Budget exhausted!\n')
            end
            break
        end
        if sum(y==0)==0
            if verbosity > 0
                fprintf('No unlabeled points left to query!\n')
            end
            break
        end
        
        % Choose query node
        probVector = get_query_probabilities_active(pruning, tree, stats);
        assert(~isempty(probVector) && all(~isnan(probVector)));
        cumProbVector = cumsum(probVector);
        sample = rand();
        v = pruning(find(cumProbVector >= sample, 1));
        assert(~isempty(v))
        
        % Add to list of selected nodes
        selectedNodes(b) = v;

        % Choose query leaf from node, then query label from oracle
        leaves = get_leaves(tree, v);
        ix = y(leaves)==0; % & w(leaves)>0;
        if sum(ix) > 1
            leaf = randsample(leaves(ix), 1);
        else
            leaf = leaves(ix);
        end
        assert(~isempty(leaf) && length(leaf)==1)
        
        y(leaf) = oracle(leaf);
        query_num = query_num + 1;
        queries = [ queries leaf ];
%         quer_hist(query_num) = leaf;
%         w = weight_fn(X, tree_children, y, params);
        
        % Update node statistics
        stats = bottom_up_update(tree, stats, leaf, v, y);
        
        if verbosity > 2
            fprintf('Query %4d: node %4d leaf %4d label %4d\n', query_num, v, c, y(leaf))
        end
    end
    
    quer_hist{batch_num+1} = queries';
    
    % Update pruning
    pruning = prune_tree(pruning, unique(selectedNodes(selectedNodes>0)), tree, stats.canSplitNode(:,1));
    prunings{batch_num+1} = pruning;

    % Impute labels
    yimp(:,batch_num+1) = impute_labels(pruning, tree, stats.labelCounts(:,:,1), stats.bestLabel(:,1));

    if query_num >= budget || sum(y==0) == 0
        break
    end
end
if verbosity > 0
    fprintf('\n')
end

y = unmap_labels(y, labels);
for i=1:size(yimp,2)
    yimp(:,i) = unmap_labels(yimp(:,i), labels);
end

end
