function [ h, hp ] = knn(X, y, w, params)
% KNN(X, y, w, ~) KNN base learner subroutine for
% active learning and active transfer learning. Uses a "hack" to learn
% the "alternative" hypothesis for IWCAL and TIWCAL.
%
% INPUT
%   X           N x D matrix of features
%   y           N x 1 vector of labels (1 positive, -1 negative)
%   w           N x 1 vector of instance weights; inf weight => constraint
%   PARAMS:             struct with additional named parameters
%     k                 number of nearest neighbors
%
% RETURNS
%   h, hp       hypothesis, i.e., function that makes predictions given
%               data: [ yhat, ydv ] = h(X)
% 
% The primary purpose of this function is to support the synthetic data
% experiments in our SDM 2015 paper on "Hierarchical Active Transfer
% Learning." It adheres to the same interface as the other base learners
% but in fact returns no alternative hypothesis and so is not compatible
% with the IWCAL AND TIWCAL algorithms.
%
% AUTHOR:   Dave Kale (dkale@usc.edu), USC
% DATE:     2015-01-26

if isfield(params, 'k')
    k = params.k;
else
    k = 1;
end

h = model_wrapper(X, y, w, k);
hp = [];

end


%%%%%


%% ERM hypothesis wrapper
function h = model_wrapper(X, y, ~, k)

h = @predict;
    
    function [ yhat, ydv ] = predict(Xnew)
        yhat = zeros(size(Xnew,1),1);
        ydv = zeros(size(Xnew,1),1);
        for i=1:size(Xnew,1)
            [d,idx] = min(sum(bsxfun(@minus, X, Xnew(i,:)).^2,2),[],1);
            yhat(i) = y(idx);
            ydv(i) = d;
        end
    end

end
