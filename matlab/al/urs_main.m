function [ y, quer_hist ] = urs_main(X, y, oracle, budget, batch_size, ...
                                     base_learner, params)
% URS_MAIN(X, Y, ORACLE, BUDGET, BATCH_SIZE, BASE_LEARNER, PARAMS)
% Aggressive (greedy) active learning based on uncertainty region sampling.
%
% INPUT
%   X               N x P data design matrix
%   y               N x 1 vector of available labels; NaN indicates
%                   unlabeled point.
%   oracle          N x 1 vector of ALL labels, used as oracle
%   budget          query budget
%   batch_size      number of queries to make for each batch
%   base_learner    learner from base-learners directory
%   params          struct of optional parameters to control behavior of
%                   URS and base learner.
%       .w          instance weights
%
% RETURNS
%   y               vector of available labels after active learning
%   quer_hist       cell array of queries made for each batch
%
% Queries most "uncertain" data point. While it hypothetically works with
% any base learner, the way it chooses the query point is currently
% hard-coded to work with max margin learners (e.g., SVM) by querying the
% point with the smallest absolute value decision value. That will not work
% for logistic regression, obviously.
%
% AUTHOR:   David Kale (dkale@usc.edu)
% DATE:     2015-01-26

if isfield(params, 'w')
    w = params.w;
else
    if isfield(params, 'w')
        w = params.w;
    else
        w = ones(size(y));
    end
end
assert(all(size(w) == size(y)));

n_batches = ceil(budget / batch_size);
batch_num = 1;
num_queries = 0;

quer_hist = cell(1,n_batches+1);
quer_hist{1} = [];

while num_queries < budget && any(isnan(y))
    uix = isnan(y);
    lix = ~uix;
    uno = find(uix);
    if length(unique(y(lix)))<=1
        qno = randsample(uno, min(batch_size, length(uno)));
        quer_hist{batch_num+1} = qno;
    else
        h = base_learner(X(lix,:), y(lix), w(lix), params);
        [~,ydv] = h(X(uix,:));
        [~,suno] = sort(abs(ydv), 'ascend');
        qno = uno(suno(1:min(batch_size, length(suno))));
        quer_hist{batch_num+1} = qno;
    end
    y(quer_hist{batch_num+1}) = oracle(quer_hist{batch_num+1});
    num_queries = num_queries + length(quer_hist{batch_num+1});
    batch_num = batch_num + 1;
end

end
