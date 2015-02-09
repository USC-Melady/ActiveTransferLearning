function [Xt] = generate_1d_synthetic_data_Target(N)
% GENERATE_1D_SYNTHETIC_DATA_TARGET(N) Generates random 1D Target data set
% from synthetic data experiments in our SDM 2015 paper, "Hierarchical
% Active Transfer Learning."
%
% INPUT
%   N           Number of points to generate
%
% RETURNS
%   Xt          N-point 1D target data set
% 
% AUTHOR:   Dave Kale (dkale@usc.edu), USC
% DATE:     2015-01-26

assert(mod(N,4) == 0)
nc = N / 4;

% cluster 1
Xt = [ rand(nc,1) ];
% cluster 2
Xt = [ Xt; rand(nc,1) + 1.5 ];
% cluster 3
Xt = [ Xt; rand(floor(nc),1) + 3 ];
% cluster 4
Xt = [ Xt; rand(floor(nc),1) + 4.5 ];

end
