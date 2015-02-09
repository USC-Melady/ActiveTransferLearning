function plot_deviations_for_figure1(dev_real, dev_predicted, labels, varargin)
% PLOT_DEVIATIONS_FOR_FIGURE1(dev_real, dev_predicted, labels, varargin)
% Generates deviation bound plot in Figure 1(c) from our SDM 2015 paper,
% "Hierarchical Active Transfers Leanring."
%
% INPUT
%   dev_real        (average) empirical deviations
%   dev_predicted   predicted (approximate) deviations from bounds
%   labels          labels for each source (cell array)
% (optional)
%   h           optional figure handle
% 
% AUTHOR:   Dave Kale (dkale@usc.edu), USC
% DATE:     2015-01-26

if nargin > 3
    h = varargin{1};
else
    h = figure;
end
clf(h);
figure(h);
hold on;

N = size(dev_real,1);
errorbar_groups([ mean(dev_real,1); mean(dev_predicted,1) ], ...
           1.96*[ std(dev_real,1); std(dev_predicted,1) ]/sqrt(N))

set(gca, 'YTickLabel', get(gca, 'YTickLabel'), 'FontSize', 14);
set(gca, 'XTickLabel', labels, 'FontSize', 14);
hl = legend('|\epsilon_T(P)-(\epsilon(Q)+\eta)|', ...
            '(1-\alpha)/2 d_{H \Delta H}', 'Location', 'Best');
set(hl, 'FontSize', 14);
title('Tightness of Theorem 1 bound', 'FontSize', 18)
xlabel('Source data', 'FontSize', 18)
ylabel('Deviation', 'FontSize', 18)

end
