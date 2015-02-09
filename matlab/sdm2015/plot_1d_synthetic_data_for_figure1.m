function plot_1d_synthetic_data_for_figure1(Xt, yt, Xs, ys, dA, labels, ...
                                            colors, varargin)
% PLOT_1D_SYNTHETIC_DATA_FOR_FIGURE1(N) Generates Figure 1(a) from our SDM
% 2015 paper, "Hierarchical Active Transfers Leanring."
%
% INPUT
%   Xt, yt      1D target data and labels
%   Xs          1D source data: N points x S sources
%   ys          1D source labels: N points x S sources
%   dA          dA distances, one for each source
%   labels      labels for each source (cell array)
% (optional)
%   h           optional figure handle
% 
% AUTHOR:   Dave Kale (dkale@usc.edu), USC
% DATE:     2015-01-26

if nargin > 7
    h = varargin{1};
else
    h = figure;
end
clf(h);
figure(h);
hold on;

al = 0.25;

plot(Xt(yt(:,1)<0), 0.5, 'b.', 'MarkerSize', 20);
plot(Xt(yt(:,1)>0), 0.5, 'r.', 'MarkerSize', 20);
ylabels = { 'Target' };
[~,ix] = sort(dA);
for si=1:numel(dA)
    X = Xs(:,ix(si));
    y = ys(:,ix(si));
    plot(X(y<0), dA(ix(si)), 'b.', 'MarkerSize', 20);
    plot(X(y>0), dA(ix(si)), 'r.', 'MarkerSize', 20);
    ylabels = { ylabels{:} sprintf('%s: %.4f', labels{ix(si)}, dA(ix(si))) };
end
minx = min(min([Xt Xs]));
maxx = max(max([Xt Xs]));
fill([ minx maxx maxx minx ], [ 0.495 0.495 0.505 0.505 ], 'k', 'EdgeColor', 'k', 'FaceAlpha', al, 'EdgeAlpha', al);
for si=1:numel(dA)
    fill([ minx maxx maxx minx ], [ dA(si)-0.005 dA(si)-0.005 dA(si)+0.005 dA(si)+0.005 ], ...
           colors(mod(si-1,length(colors))+1), 'EdgeColor', colors(mod(si-1,length(colors))+1), 'FaceAlpha', al, 'EdgeAlpha', al);
end
set(gca, 'YTick', sort([ 0 dA ]), 'FontSize', 14);

set(gca, 'YTickLabel', ylabels, 'FontSize', 14);
ylabel('Approx. d_{H \Delta H} distance', 'FontSize', 18)
xlabel('Data', 'FontSize', 18)
title('1D synthetic data sets', 'FontSize', 18)
ylim([0.495 ceil(max(dA)*10)/10])
xlim([minx maxx])

end
