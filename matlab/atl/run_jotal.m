function [ perf, misc ] = run_jotal(Xs, ys, ds, Xtr, ytr, budget, ...
                                    batch_size, base_learner, Xte, ytre, ...
                                    params)

Xst = [Xs;Xtr];
yst = [ys;ytr];

if ~isfield(params, 'kernel')
    params.kernel = @gaussian_kernel;
    params.kparam = 1/size(Xs,2);
else
    if ~isfield(params, 'kparam')
        params.kparam = 1/size(Xs,2);
    end
end

params.dom = [ ds;zeros(size(ytr)) ];
[ yq, quer_hist, b_hist, wq ] = jotal_main(Xst, [ys; nan(size(ytr))], ...
                                           yst, budget, batch_size, params);

[kmmd_fn] = make_kmmd_approximator(params.kernel, params.kparam);

perf = struct();
lix = [ true(size(ys)); false(size(ytr)) ];
nq = 0;
for bi=1:numel(quer_hist)
    nq = nq + length(quer_hist{bi});
    perf(bi).num_quer = nq;
    
    lix(quer_hist{bi}) = true;
    if isempty(quer_hist{bi})
        [bs_mn,~,~] = solve_jotal_qp(Xs, [], Xtr, 0, params.kernel, params);
        bs_sm = bs_mn;
    else
        bs_mn = mean([ b_hist{1:bi} ],2);
        bs_sm = sum([ b_hist{1:bi} ],2);
    end
    
    bst_mn = [ bs_mn; ones(size(ytr)) ];
    bst_sm = [ bs_sm; ones(size(ytr)) ];
    if any(~lix)
        misc(bi).kmmd_u_mn = kmmd_fn(Xst(lix,:), Xst(~lix,:), bst_mn(lix));
        misc(bi).kmmd_u_sm = kmmd_fn(Xst(lix,:), Xst(~lix,:), bst_sm(lix));
    else
        misc(bi).kmmd_u_mn = 0;
        misc(bi).kmmd_u_sm = 0;
    end
    misc(bi).kmmd_t_mn = kmmd_fn(Xst(lix,:), Xtr, bst_mn(lix));    
    misc(bi).kmmd_t_sm = kmmd_fn(Xst(lix,:), Xtr, bst_sm(lix));
    
    yl = yst(lix);
    if isempty(yl) || length(unique(yl))<=1
        perf(bi).prec_at_90 = 0.5;
        perf(bi).train = generate_dummy_metrics(yst);
        perf(bi).test  = generate_dummy_metrics(ytre);
        
        perf(bi).prec_at_90_2 = 0.5;
        perf(bi).train_2 = generate_dummy_metrics(yst);
        perf(bi).test_2  = generate_dummy_metrics(ytre);
    else
        h = base_learner(Xst(lix,:), yl, bst_mn(lix), params);
        [yh,ydv] = h(Xst(lix,:));
        perf(bi).train = compute_fave_metrics(yl,yh,ydv);
        [yh,ydv] = h(Xte);
        perf(bi).test = compute_fave_metrics(ytre,yh,ydv);
        [tpr,~,ppv] = prc_stats_binormal(ytre,ydv,true);
        perf(bi).prec_at_90 = ppv(max(find(tpr<0.9,1)-1,1));
        
        h = base_learner(Xst(lix,:), yl, bst_sm(lix), params);
        [yh,ydv] = h(Xst(lix,:));
        perf(bi).train_2 = compute_fave_metrics(yl,yh,ydv);
        [yh,ydv] = h(Xte);
        perf(bi).test_2 = compute_fave_metrics(ytre,yh,ydv);
        [tpr,~,ppv] = prc_stats_binormal(ytre,ydv,true);
        perf(bi).prec_at_90_2 = ppv(max(find(tpr<0.9,1)-1,1));
    end
end

end
