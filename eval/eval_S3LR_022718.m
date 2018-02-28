function eval_S3LR_022718(prefix,n,d,D,Ng,sigma,rho,seed,lambda,Gamma,lambdaIncr,jobIdx)

% lambda, Gamma, gamma0, relax, affine, t_max, opt, T, nu1, nu2, tau,
% lambda_max, Gamma_max)

rng(seed);
[X, groupsTrue, Omega] = generate_scmd_data_matrix(n,d,D,Ng,sigma,rho,1,seed);
Xnorm = sqrt(sum(X.^2));
X = X ./ repmat(Xnorm, [D 1]);
Xobs = X.*Omega;

form = 'S3LR';
gamma0 = Gamma;
maxIter = 300;

% Copy defaults from S3LR
relax = 1; affine = 0;
opt.tol =1e-4;
opt.maxIter =1e6;
opt.rho =1.1;
opt.mu_max =1e4;
opt.norm_sr ='1';
opt.norm_mc ='1';
T = 1;

[~, ~, C, Y, ~] = S3LR(Xobs, Omega, groupsTrue, X, lambda, Gamma, gamma0, ...
    relax, affine, maxIter, opt, T, lambdaIncr);

A = build_affinity(C); C = sparse(C);
groups = spectral_clustering(A, n);
[cluster_err, groups] = eval_cluster_error(groups, groupsTrue);

Xunobs = X(~Omega);
comp_err = norm(Y(~Omega) - Xunobs)/norm(Xunobs);

true_scores = [comp_err cluster_err];
fprintf('cmperr=%.3f, clstrerr=%.3f\n', comp_err, cluster_err);

fname = sprintf('%s_%s_n%d_d%d_D%d_Ng%d_sigma%.0e_rho%.1f_seed%d_lamb%.2e_Gamma%.2e_lambIncr%.2f_job%05d.mat', ...
    prefix, form, n, d, D, Ng, sigma, rho, seed, lambda, Gamma, lambdaIncr, jobIdx);

save(fname, 'form', 'n', 'd', 'D', 'Ng', 'sigma', 'rho', 'seed', 'lambda', 'Gamma', 'lambdaIncr', ...
    'true_scores', 'jobIdx');
end
