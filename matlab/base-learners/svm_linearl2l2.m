function [ h, hp ] = svm_linearl2l2(X, y, w, params)
% SVM_LINEARL2L2(X, y, w, params) Linear SVM base learner subroutine for
% active learning and active transfer learning. Uses both "retraining" and
% a "hack" to learn the "alternative" hypothesis for TIWCAL.
%
% INPUT
%   X                   N x D matrix of features
%   y                   N x 1 vector of labels (1 positive, -1 negative)
%   w                   N x 1 vector of instance weights; inf => constraint
%   PARAMS:             struct with additional named parameters
%     cost              classifier cost, see liblinear docs (default 1)
%     constraintWtFn    function to compute constraint point weight
%     adjustThreshold   adjust the threshold to ensure disagreement
%     gamma             gamma for RBF kernel
%     shrinking         whether to use shrinking heuristic
%
% RETURNS
%   h, hp       two hypotheses, i.e., functions that make predictions given
%               data: [ yhat, ydv ] = h(X). One is the ERM, the other the
%               alternative hypothesis that disagrees with the ERM.
% 
% These were originally written to be used with our implementation of the
% TIWCAL active transfer learning algorithm from our ICDM 2013 paper,
% "Accelerating Active Learning with Transfer Learning." It's a
% wrapper around the weighted liblinear classifier that (1) provides a generic
% interface so we can try different learners and (2) implements step 2 from
% Figure 1 of the original IWALCAL paper, where two hypotheses must be trained
% on a  data set but constrained to disagree on an additional point. Of course,
% such a constraint can be difficult or impossible to implement, so this
% learner employs heuristics.
%
% Here we train a weighted instance linear SVM using the liblinear library
% modified to accept instance weights
% (http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/#weights_for_data_instances).
% When MEXing liblinear, be sure to rename the train and predict functions
% to lltrain and llpredict. To obtain the "alternative" hypothesis, we (1)
% predict the label of the constraint point with the first hypothesis,
% (2) flip its label, (3) assign it a really high weight (so the learner
% will strongly prefer to get it correct), and (4) train a new hypothesis.
% However we also calculate by how much we must shift the decision value of
% a prediction to flip the label of the constraint point and then
% apply that shift to every prediction. The constraint is indicated using
% an infinitely large weight. For now, this is hard wired to allow at most
% one constraint point.
%
% AUTHOR:   Dave Kale (dkale@usc.edu), USC
% DATE:     2015-01-26

if isempty(w)
    w = ones(size(y));
end

%%
idx   = ~isinf(w);          % Indeces of non-constraint training set
assert(sum(~idx) <= 1);     % <=1 constraint points allowed
X = sparse(X);              % liblinear requires sparse

if ~isfield(params, 'cost')
    params.cost = 1;
end

if isfield(params, 'constraintWtFn')
    constraintWtFn = params.constraintWtFn;
else
    constraintWtFn = @(w) 100 * sum(w(~isinf(w)));
end

if isfield(params, 'adjustThreshold')
    adjustThreshold = params.adjustThreshold;
else
    adjustThreshold = 1;
end

% use primal formulation if N > D
if size(X,1) > size(X,2)
    svmopt = sprintf('-s 2 -c %f -B 1 -q', params.cost);
else % else use dual
    svmopt = sprintf('-s 1 -c %f -B 1 -q', params.cost);
end

% ERM hypothesis
model = lltrain(w(idx), y(idx), X(idx,:), svmopt);
h = model_wrapper(model);

% Alternative hypothesis
if sum(~idx) == 1
    hp = model_wrapper_alt(model, svmopt, X, y, w, constraintWtFn, adjustThreshold);
    [ oldy, ~ ] = h(X(~idx,:));
    [ newy, ~ ] = hp(X(~idx,:));
    if adjustThreshold
        assert(newy ~= oldy)    % this shouldn't happen
    end
else
    hp = [];
end

end


%%%%%


%% ERM hypothesis wrapper
function h = model_wrapper(model)

h = @predict;
    
    function [ yhat, ydv ] = predict(X)
        [ yhat, ~, ydv ] = llpredict(zeros(size(X,1),1), sparse(X), model, '-q');
        if model.Label(1) < 0
            ydv = -ydv;
        end
    end

end

%% Alternative hypothesis wrapper
function h = model_wrapper_alt(model, svmopt, X, y, w, constraintWtFn, adjustThreshold)

h = @predict;

idx = isinf(w);     % Index of constraint point
[ dy, ~, ~ ] = llpredict(0, X(idx,:), model, '-q');

% HERE IS HACK #1: set constraint point's weight to be equal to 100 times
% sum of all the other training data weights and its "label" to be opposite
% of dy
w(idx) = constraintWtFn(w);
y(idx) = -dy;

model = lltrain(w, y, X, svmopt);
[ newy, ~, dv ] = llpredict(0, X(idx,:), model, '-q');
if model.Label(1) < 0   % required for libsvm/liblinear to ensure that
    dv = -dv;           % signs of labels, decision values match
end

% HERE IS HACK #2: if hack #1 didn't work and we're still predicting the
% "oldy" label for the constraint point, then shift the shift decision
% boundary from zero to the constraint point's decision value.
if adjustThreshold && newy==dy
    shift = newy * (abs(dv) + 1E-12);
else
    shift = 0;
end

    function [ yhat, ydv ] = predict(X)
        [ ~, ~, ydv ] = llpredict(zeros(size(X,1),1), sparse(X), model, '-q');
        if model.Label(1) < 0   % required for libsvm/liblinear to ensure
                                % that signs of labels, decision values
                                % match
            ydv = -ydv;
        end
        ydv = ydv - shift;
        yhat = sign(ydv);
        yhat(yhat==0) = -1;
    end

end
