function [Xs] = generate_1d_synthetic_data_S4(N)
% GENERATE_1D_SYNTHETIC_DATA_S1(N) Generates random 1D S4 source data set
% from synthetic data experiments in our SDM 2015 paper, "Hierarchical
% Active Transfer Learning."
%
% INPUT
%   N           Number of points to generate
%
% RETURNS
%   Xs          N-point 1D target data set
% 
% AUTHOR:   Dave Kale (dkale@usc.edu), USC
% DATE:     2015-01-26

assert(mod(N,4) == 0)
nc = N / 4;

% cluster 1
Xs = [ rand(nc,1) - 1.5 ];
% cluster 2
Xs = [ Xs; rand(nc,1) + 1.5 ];
% cluster 3
Xs = [ Xs; rand(floor(nc),1) + 3 ];
% cluster 4
Xs = [ Xs; rand(floor(nc),1) + 6 ];

end
