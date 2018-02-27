function eval_S3LR_022618(prefix,n,d,D,Ng,rho,seed,lambda,Gamma,gamma0)

% lambda, Gamma, gamma0, relax, affine, t_max, opt, T, nu1, nu2, tau,
% lambda_max, Gamma_max)

rng(seed);
[X, groupsTrue, Omega] = generate_scmd_data_matrix(n,d,D,Ng,0,rho,1,seed);
Xnorm = sqrt(sum(X.^2));
X = X ./ repmat(Xnorm, [D 1]);
Xobs = X.*Omega;

form = 'S3LR';
[~, ~, C, Y, ~] = S3LR(Xobs, Omega, groupsTrue, X, lambda, Gamma, gamma0);


A = build_affinity(C);
groups = spectral_clustering(A, n);
[cluster_err, groups] = eval_cluster_error(groups, groupsTrue);

Xunobs = X(~Omega);
comp_err = norm(Y(~Omega) - Xunobs)/norm(Xunobs);

true_scores = [comp_err cluster_err];
fprintf('cmperr=%.3f, clstrerr=%.3f\n', comp_err, cluster_err);

fname = sprintf('%s_%s_n%d_d%d_D%d_Ng%d_rho%.1f_seed%d_lamb%.2e_alpha%.2e.mat', ...
    prefix, form, n, d, D, Ng, rho, seed, lambda, Gamma);

save(fname, 'form', 'n', 'd', 'D', 'Ng', 'rho', 'seed', 'lambda', 'Gamma', ...
    'groups', 'C', 'Y', 'true_scores');
end
