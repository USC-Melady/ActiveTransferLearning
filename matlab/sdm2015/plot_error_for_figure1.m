function plot_error_for_figure1(P_hsal, P_hatl, labels, colors, varargin)
% PLOT_ERROR_FOR_FOR_FIGURE1(P_hsal, P_hatl, labels, colors, varargin)
% Generates 1D synthetic data error plots (for, e.g., Figures 1(b,d)) from
% our SDM 2015 paper, "Hierarchical Active Transfers Leanring."
%
% INPUT
%   P_hsal      performance data for HSAL: 1 x (#batches) x (#reps) x 3
%               array. The last dimension includes number of queries,
%               target-only label imputation error, and overall label
%               imputation error, in that order.
%   P_hatl      performance data for HATL for all sources: (#sources) x 
%               (#batches) x (#reps) x 3 array. The last dimension includes
%               number of queries, target-only label imputation error, and
%               overall label imputation error, in that order.
%   labels      labels for each source (cell array)
%   colors      string (character array) of colors
% (optional)
%   h           optional figure handle
% 
% AUTHOR:   Dave Kale (dkale@usc.edu), USC
% DATE:     2015-01-26

if nargin > 4
    h = varargin{1};
else
    h = figure;
end
clf(h);
figure(h);
hold on;

if ~isempty(P_hsal)
    avgP_hsal  = squeeze(mean(P_hsal,3));
    stdP_hsal  = 1.96*squeeze(std(P_hsal,[],3)) / sqrt(size(P_hsal,3));
    fill([avgP_hsal(:,1); flipud(avgP_hsal(:,1))], [avgP_hsal(:,2)-stdP_hsal(:,2); flipud(avgP_hsal(:,2)+stdP_hsal(:,2))], 'k', 'EdgeColor', 'k', 'EdgeAlpha', 0.25, 'FaceAlpha', 0.25)
end

avgP_hatl = squeeze(mean(P_hatl, 3));
stdP_hatl = 1.96*squeeze(std(P_hatl,[],3)) / sqrt(size(P_hatl,3));
for si=1:size(avgP_hatl,1)
    fill([avgP_hatl(si,:,1) fliplr(avgP_hatl(si,:,1))], [avgP_hatl(si,:,2)-stdP_hatl(si,:,2) fliplr(avgP_hatl(si,:,2)+stdP_hatl(si,:,2))], ...
         colors(mod(si-1,length(colors))+1), 'EdgeColor', colors(mod(si-1,length(colors))+1), 'EdgeAlpha', 0.25, 'FaceAlpha', 0.25)
end

hs = [];
llabels = {};
if ~isempty(P_hsal)
    hh = plot(avgP_hsal(:,1), avgP_hsal(:,2), 'k-', 'LineWidth', 2);
    hs = [hs hh];
    llabels = { llabels{:} 'HSAL' };
end

for si=1:size(avgP_hatl,1)
    hh = plot(avgP_hatl(si,:,1), avgP_hatl(si,:,2), [colors(mod(si-1,length(colors))+1) '-'], 'LineWidth', 2);
    hs = [ hs hh ];
    llabels = { llabels{:} sprintf('HATL: %s', labels{si}) };
end

ylim([0 0.5])
% set(gca, 'XTick', 0:5:50, 'FontSize', 14);
% set(gca, 'YTickLabel', get(gca, 'YTickLabel'), 'FontSize', 18);
title('Target label imputation error', 'FontSize', 18)
xlabel('Target label queries', 'FontSize', 18)
ylabel('\epsilon_T(P)', 'FontSize', 18)

hl = legend(hs, llabels, 'Location', 'Best');
set(hl, 'FontSize', 14);

end
