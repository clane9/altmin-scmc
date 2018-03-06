function eval_ENSC_MC_methods_030518(prefix,form,n,d,D,Ng,sigma,rho,seed, ...
    initMode,lambda,gamma,tauScheme,lambdaIncr,maxIter,maxSecs,jobIdx)

rng(seed);
[X, groupsTrue, Omega] = generate_scmd_data_matrix(n,d,D,Ng,sigma,rho,1,seed);
Xnorm = sqrt(sum(X.^2));
X = X ./ repmat(Xnorm, [D 1]);
Xobs = X.*Omega;

if strcmpi(form, 'ENSC_MC')
  solver = ENSC_MC_spams_comp_apg(Xobs, Omega, n, lambda, gamma, 0);
else
  error('formulation not implemented.')
end

opt_params.maxIter = maxIter; opt_params.convThr = 1e-6; % opt_params.convThr = -Inf;
opt_params.lambdaIncr = lambdaIncr;
opt_params.tauScheme = tauScheme;
opt_params.trueData = {X, groupsTrue};
opt_params.prtLevel = 1; opt_params.logLevel = 3;
exprC_params = struct;
compY_params.maxIter = 500; compY_params.convThr = 1e-4; % APG parameters

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
opt_params.maxTime = maxSecs - toc;

[~, ~, ~, history] = solver.solve(Y0, opt_params, exprC_params, compY_params);

fname = sprintf('%s_%s_n%d_d%d_D%d_Ng%d_sigma%.0e_rho%.2f_seed%d_init%s_lamb%.1e_gamma%.1f_tauScheme%d-%d_lambIncr%.2f_job%05d.mat', ...
    prefix, form, n, d, D, Ng, sigma, rho, seed, initMode, lambda, gamma, tauScheme, lambdaIncr, jobIdx);

opt_params.trueData = {};
save(fname, 'form', 'n', 'd', 'D', 'Ng', 'sigma', 'rho', 'seed', 'initMode', 'lambda', 'gamma', ...
    'opt_params', 'init_params', 'history', 'jobIdx');
end
