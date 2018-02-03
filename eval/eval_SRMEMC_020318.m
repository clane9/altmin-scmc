function eval_SRMEMC_020318(prefix,n,d,D,Ng,rho,seed,lambda,alpha)

rng(seed);
[X, groupsTrue, Omega] = generate_scmd_data_matrix(n,d,D,Ng,0,rho,1,seed);
Xnorm = sqrt(sum(X.^2));
X = X ./ repmat(Xnorm, [D 1]);
Xobs = X.*Omega;

form = 'SRME_MC';
[groups, C, Y, history] = SRME_MC_ADMM_ron(Xobs,Omega,n,lambda,alpha);

Xunobs = X(~Omega);
comp_err = norm(Y(~Omega) - Xunobs)/norm(Xunobs);
[cluster_err, groups] = eval_cluster_error(groups, groupsTrue);
history.true_scores = [comp_err cluster_err];

fname = sprintf('%s_%s_n%d_d%d_D%d_Ng%d_rho%.1f_seed%d_lamb%.2e_alpha%.2e.mat', ...
    prefix, form, n, d, D, Ng, rho, seed, lambda, alpha);

save(fname, 'form', 'n', 'd', 'D', 'Ng', 'rho', 'seed', 'lambda', 'alpha', ...
    'groups', 'C', 'Y', 'history');
end
