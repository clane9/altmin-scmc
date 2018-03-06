function eval_S3LR_030518(prefix,n,d,D,Ng,sigma,rho,seed,initMode,...
    lambda,Gamma,lambdaIncr,maxIter,maxSecs,jobIdx)

rng(seed);
[X, groupsTrue, Omega] = generate_scmd_data_matrix(n,d,D,Ng,sigma,rho,1,seed);
Xnorm = sqrt(sum(X.^2));
X = X ./ repmat(Xnorm, [D 1]);
Xobs = X.*Omega;

form = 'S3LR';
gamma0 = Gamma;

% Copy defaults from S3LR
relax = 1; affine = 0;
opt.tol =1e-4;
opt.maxIter =1e6;
opt.rho =1.1;
opt.mu_max =1e4;
opt.norm_sr ='1';
opt.norm_mc ='1';
T = 1;

tic;
init_params.lambda = 20;
init_params.tauScheme = [inf 0];
init_params.lambdaIncr = 1.1;
init_params.maxIter = 10;
if strcmpi(initMode, 'pzf_ssc')
  tmp_solver = ENSC_Group_MC_spams(Xobs, Omega, n, init_params.lambda, 1);
  tmp_opt_params = struct('maxIter', 1, 'tauScheme', init_params.tauScheme, ...
      'prtLevel', 0, 'logLevel', 0);
  [~,~,Y0,~] = tmp_solver.solve([], tmp_opt_params);
elseif strcmpi(initMode, 'alt_pzf_ssc')
  tmp_solver = ENSC_Group_MC_spams(Xobs, Omega, n, init_params.lambda, 1);
  tmp_opt_params = struct('maxIter', init_params.maxIter, 'tauScheme', init_params.tauScheme, ...
      'lambdaIncr', init_params.lambdaIncr, 'prtLevel', 0, 'logLevel', 0);
  [~,~,Y0,~] = tmp_solver.solve([], tmp_opt_params);
else
  % default: zero-filling.
  Y0 = [];
end
maxSecs = maxSecs - toc;

% function [acc_i, mc_err_i, Z, X_fill, iterStatus] =S3LR(X, Omega, idx, X0,
%    lambda, Gamma, X_fill0, gamma0, relax, affine, t_max, maxTime, opt, T, nu1, nu2,
%    tau, lambda_max, Gamma_max)
[~, ~, C, Y, ~] = S3LR(Xobs, Omega, groupsTrue, X, lambda, Gamma, Y0, gamma0, ...
    relax, affine, maxIter, maxSecs, opt, T, lambdaIncr);

A = build_affinity(C);
groups = spectral_clustering(A, n);
[cluster_err, ~] = eval_cluster_error(groups, groupsTrue);

Xunobs = X(~Omega);
comp_err = norm(Y(~Omega) - Xunobs)/norm(Xunobs);

true_scores = [comp_err cluster_err];
fprintf('cmperr=%.3f, clstrerr=%.3f\n', true_scores);

fname = sprintf('%s_%s_n%d_d%d_D%d_Ng%d_sigma%.0e_rho%.2f_seed%d_init%s_lamb%.2e_Gamma%.2e_lambIncr%.2f_job%05d.mat', ...
    prefix, form, n, d, D, Ng, sigma, rho, seed, initMode, lambda, Gamma, lambdaIncr, jobIdx);

save(fname, 'form', 'n', 'd', 'D', 'Ng', 'sigma', 'rho', 'seed', ...
    'initMode', 'lambda', 'Gamma', 'lambdaIncr', 'true_scores', 'jobIdx');
end
