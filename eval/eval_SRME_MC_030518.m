function eval_SRME_MC_030518(prefix,n,d,D,Ng,sigma,rho,seed,initMode,...
    lambda,alpha,maxIter,maxSecs,jobIdx)

rng(seed);
[X, groupsTrue, Omega] = generate_scmd_data_matrix(n,d,D,Ng,sigma,rho,1,seed);
Xnorm = sqrt(sum(X.^2));
X = X ./ repmat(Xnorm, [D 1]);
Xobs = X.*Omega;

form = 'SRME_MC';

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

% function [X,Z,E]=SRME_MC_ADMM(X0,M,Xinit,lambda,alpha,params)
opt_params.maxIter = maxIter;
opt_params.convThr = 1e-6; % opt_params.convThr = -Inf;
opt_params.maxTime = maxSecs - toc;
[Y, C, ~] = SRME_MC_ADMM(Xobs, Omega, Y0, lambda, alpha, opt_params);

A = build_affinity(C);
groups = spectral_clustering(A, n);
[cluster_err, ~] = eval_cluster_error(groups, groupsTrue);

Xunobs = X(~Omega);
comp_err = norm(Y(~Omega) - Xunobs)/norm(Xunobs);

true_scores = [comp_err cluster_err];
fprintf('cmperr=%.3f, clstrerr=%.3f\n', true_scores);

fname = sprintf('%s_%s_n%d_d%d_D%d_Ng%d_sigma%.0e_rho%.2f_seed%d_init%s_lamb%.2e_alpha%.2e_job%05d.mat', ...
    prefix, form, n, d, D, Ng, sigma, rho, seed, initMode, lambda, alpha, jobIdx);

save(fname, 'form', 'n', 'd', 'D', 'Ng', 'sigma', 'rho', 'seed', 'lambda', 'alpha', ...
    'initMode', 'opt_params', 'true_scores', 'jobIdx');
end
