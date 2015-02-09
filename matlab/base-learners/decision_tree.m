function [ h, hp ] = decision_tree(X, y, w, ~)
% DECISION_TREE(X, y, w, ~) Decision Tree base learner subroutine for
% active learning and active transfer learning. Uses a "hack" to learn
% the "alternative" hypothesis for IWCAL and TIWCAL.
%
% INPUT
%   X           N x D matrix of features
%   y           N x 1 vector of labels (1 positive, -1 negative)
%   w           N x 1 vector of instance weights; inf weight => constraint
%
% RETURNS
%   h, hp       hypothesis, i.e., function that makes predictions given
%               data: [ yhat, ydv ] = h(X)
% 
% These were originally written to be used with our implementation of the
% TIWCAL active transfer learning algorithm from our ICDM 2013 paper,
% "Accelerating Active Learning with Transfer Learning." It's a
% wrapper that (1) provides a generic interface so we can try different
% learners and (2) implements step 2 from Figure 1 of the original IWCAL
% paper, where two hypotheses must be trained on a  data set but constrained
% to disagree on an additional point. Of course, such a constraint can be
% difficult or impossible to implement, so this learner employs heuristics.
%
% Here we train a weighted instance decision tree using the built-in MATLAB
% routine, and we obtain the "alternative" hypothesis by flipping the label
% of any data point that reaches the same node as the constraint
% point. The constraint is indicated using an infinitely large weight. For
% now, this is hard wired to allow at most one constraint point.
%
% AUTHOR:   Dave Kale (dkale@usc.edu), USC
% DATE:     2015-01-26

%%
idx = ~isinf(w);        % Indeces of non-constraint training set
assert(sum(~idx) <= 1); % <=1 constraint points allowed
X = full(X);            % MATLAB decision tree requires non-sparse

% ERM hypothesis
model = ClassificationTree.fit(X(idx,:), y(idx), 'weights', w(idx));
h = model_wrapper(model);

% Alternative hypothesis
if sum(~idx) == 1
    hp = model_wrapper_alt(model, X(~idx,:));
    [ oldy, ~ ] = h(X(~idx,:));
    [ newy, ~ ] = hp(X(~idx,:));
    assert(newy ~= oldy)    % this should never happen!
else
    hp = [];
end

end


%%%%%


%% ERM hypothesis wrapper
function h = model_wrapper(model)

h = @predict;
    
    function [ yhat, ydv ] = predict(X)
        [ yhat, yprob ] = model.predict(full(X));
        ydv = yprob(:,model.ClassNames==1);
        ydv(ydv<1E-12) = 1E-12;
        ydv(ydv>1-1E-12) = 1-1E-12;
    end

end

%% Alternative hypothesis wrapper
function h = model_wrapper_alt(model, dx)

h = @predict;
[ dy, ~, dleaf ] = model.predict(dx);   % ERM label, leaf

    function [ yhat, ydv ] = predict(X)
        [ yhat, yprob, ynode ] = model.predict(full(X));
        ydv = yprob(:,model.ClassNames==1);
        ydv(ydv<1E-12) = 1E-12;
        ydv(ydv>1-1E-12) = 1-1E-12;
        
        % HERE IS OUR HACK: flip labels if same leaf as dx
        yhat(ynode==dleaf) = -dy;
        % NOTE: probably also need to modify ydv!
    end

end
