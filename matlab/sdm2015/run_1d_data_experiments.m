close all;
clear;
clc;

N = 200;
n_reps = 10; %25;
n_srcs = 5;
batch_size = 1;
params = struct('eta', 0.05, 'delta', 0.1);

if exist('1d_synthetic_data.mat', 'file') <= 0
    fprintf('Generating 1d synthetic data...')
    Target = struct();
    Sources = cell(1,1);
    Test = struct();
    for si=1:n_srcs
        Sources{si} = struct();
    end
    for ri=1:n_reps
        Target(ri).X = generate_1d_synthetic_data_Target(N);
        Target(ri).y = label_1d_synthetic_data(Target(ri).X);
        for si=1:n_srcs
            Sources{si}(ri).X = eval(sprintf('generate_1d_synthetic_data_S%d(%d)', ...
                                             si, N));
            Sources{si}(ri).y = label_1d_synthetic_data(Sources{si}(ri).X);
            [~,dA] = approx_da_distance(Sources{si}(ri).X, Target(ri).X, @knn);
            Sources{si}(ri).dAdist = dA;
        end
        Test(ri).X = generate_1d_synthetic_data_Target(N);
        Test(ri).y = label_1d_synthetic_data(Target(ri).X);
    end
    fprintf('saving to 1d_synthetic_data.mat...')
    save('1d_synthetic_data.mat', 'Target', 'Sources', 'Test');
    fprintf('DONE!\n')
else
    fprintf('Loading from 1d_synthetic_data.mat...')
    load('1d_synthetic_data.mat', 'Target', 'Sources', 'Test');
    fprintf('DONE!\n')
end

allXs = nan(N, n_srcs);
allYs = nan(N, n_srcs);
dAdist = nan(n_reps,n_srcs);
labels = {};
colors = 'rcmgby';
for si=1:n_srcs
    allXs(:,si) = Sources{si}(1).X;
    allYs(:,si) = Sources{si}(1).y;
    dAdist(:,si) = [ Sources{si}.dAdist ];
    labels = { labels{:} sprintf('S%d', si) };
end
save('1d_synthetic_data.mat', 'allXs', 'allYs', 'dAdist', ...
     'labels', 'colors', '-append');

h = figure;
plot_1d_synthetic_data_for_figure1(Target(1).X, Target(1).y, allXs, allYs, ...
                                   mean(dAdist,1), labels, colors, h);
print('synth-data.eps', '-depsc')
print('synth-data.png', '-dpng')

assert(numel(Sources) == n_srcs);

err_imput_stuff = nan(n_srcs+1,N/batch_size+1,n_reps,2);
bound_stuff = nan(n_srcs,n_reps,5);
all_perfs = cell(n_srcs+1,n_reps);
all_miscs = cell(n_srcs+1,n_reps);

err_imput = nan(n_srcs+1,N/batch_size+1,n_reps,3);
Qstats = struct();
bound_analysis = nan(n_srcs,n_reps,4);
for ri=1:numel(Target)
    fprintf('Rep %d: T', ri)
    Xtr = Target(ri).X; ytr = Target(ri).y;
    Xte = Test(ri).X; yte = Test(ri).y;
    budget = size(Xtr,1);
    [perf,misc] = run_hsal(Xtr, ytr, budget, batch_size, @knn, Xte, yte, params);
    
    all_perfs{n_srcs+1,ri} = perf; all_miscs{n_srcs+1,ri} = misc;
    
    for bi=1:numel(misc.prunings)
        err_imput(n_srcs+1,bi,ri,1) = perf(bi).num_quer;
        err_imput(n_srcs+1,bi,ri,2:3) = 1-misc.prunings(bi).purity;
        err_imput_stuff(n_srcs+1,bi,ri,:) = misc.prunings(bi).err;
    end
    Qstats(n_srcs+1,ri).n_clust = length(misc.Q);
    Qstats(n_srcs+1,ri).depth = max(misc.dQ);
    Qstats(n_srcs+1,ri).err  = 1-misc.purity_Q;
    
    for si=1:numel(Sources)
        fprintf(' S%d', si)
        Xs = Sources{si}(ri).X; ys = Sources{si}(ri).y;
        [perf,misc] = run_hatl(Xs, ys, ones(size(ys)), Xtr, ytr, ...
                               budget, batch_size, @knn, ...
                               Xte, yte, params);

        all_perfs{si,ri} = perf; all_miscs{si,ri} = misc;

        for bi=1:numel(misc.prunings)
            err_imput(si,bi,ri,:) = [ perf(bi).num_quer ...
                                      1-misc.prunings(bi).purity_target ...
                                      1-misc.prunings(bi).purity ];
            err_imput_stuff(si,bi,ri,:) =  [ misc.prunings(bi).err_target ...
                                             misc.prunings(bi).err ];
        end
        Qstats(si,ri).n_clust = length(misc.Q);
        Qstats(si,ri).depth = max(misc.dQ);
        Qstats(si,ri).err  = 1-misc.purity_Q;
        
        [errP,nquer] = compute_hsal_bound(Qstats(si).err, Qstats(si).n_clust, ...
                                           Qstats(si).depth, params.eta, ...
                                           params.delta, batch_size);
        [errPt,dAterm,sqterm] = compute_hatl_bound(errP, size(Xs,1), ...
                                                   size(Xtr,1), ...
                                                   Sources{si}(ri).dAdist, ...
                                                   params.delta);
        
        ix = find(err_imput(si,:,ri,3)<=errP,1);
        bound_analysis(si,ri,:) = [ ix err_imput(si,ix,ri,2) errP dAterm ];
        bound_stuff(si,ri,:) = [ errP nquer errPt dAterm sqterm ];
    end
    fprintf('\n')
end
save('1d_synthetic_data.mat', 'err_imput', 'Qstats', ...
     'bound_analysis', '-append');
save('1d_synthetic_data_SUPP.mat', 'err_imput_stuff', 'bound_stuff', ...
     'all_perfs', 'all_miscs');

h = figure;
plot_error_for_figure1(err_imput(end,:,:,:), err_imput(1:n_srcs,:,:,:), ...
                       labels, colors, h);
print('synth-target-error.eps', '-depsc')
print('synth-target-error.png', '-dpng')

h = figure;
plot_deviations_for_figure1(abs(bound_analysis(:,:,2)-bound_analysis(:,:,3))', ...
                            bound_analysis(:,:,4)', labels);
print('synth-bound.eps', '-depsc')
print('synth-bound.png', '-dpng')
