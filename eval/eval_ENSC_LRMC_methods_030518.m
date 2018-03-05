function eval_ENSC_LRMC_methods_030518(prefix,form,n,d,D,Ng,sigma,rho,seed, ...
    lambda,gamma,tauScheme,lambdaIncr,maxIter,maxSecs,saveFactors,jobIdx)

rng(seed);
[X, groupsTrue, Omega] = generate_scmd_data_matrix(n,d,D,Ng,sigma,rho,1,seed);
Xnorm = sqrt(sum(X.^2));
X = X ./ repmat(Xnorm, [D 1]);
Xobs = X.*Omega;

if strcmpi(form, 'Alt_PZF_ENSC_LRMC')
  solver = ENSC_Group_MC_spams(Xobs, Omega, n, lambda, gamma);
elseif strcmpi(form, 'PZF_ENSC_LRMC')
  solver = ENSC_Group_MC_spams(Xobs, Omega, n, lambda, gamma);
  tauScheme = [inf 0];
  lambdaIncr = 1;
  maxIter = 1;
else
  error('formulation not implemented.')
end

opt_params.maxIter = maxIter; opt_params.convThr = 1e-6; % opt_params.convThr = -Inf;
opt_params.lambdaIncr = lambdaIncr;
opt_params.tauScheme = tauScheme;
opt_params.trueData = {X, groupsTrue};
opt_params.prtLevel = 1; opt_params.logLevel = 3;
exprC_params = struct; compY_params = struct;
% compY_params.maxIter = 2000; compY_params.convThr = 1e-8;
opt_params.maxTime = maxSecs;

% zero-filling initialization.
Y0 = [];

[groups, C, Y, history] = solver.solve(Y0, opt_params, exprC_params, compY_params);
C = sparse(C);

if ~saveFactors
  Y = []; C = [];
end

fname = sprintf('%s_%s_n%d_d%d_D%d_Ng%d_sigma%.0e_rho%.2f_seed%d_lamb%.1e_gamma%.1f_tauScheme%d-%d_lambIncr%.2f_job%05d.mat', ...
    prefix, form, n, d, D, Ng, sigma, rho, seed, lambda, gamma, tauScheme, lambdaIncr, jobIdx);

opt_params.trueData = {};
save(fname, 'form', 'n', 'd', 'D', 'Ng', 'sigma', 'rho', 'seed', 'lambda', 'gamma', ...
    'opt_params', 'Y', 'C', 'history', 'jobIdx');
end
