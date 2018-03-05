function eval_ENSC_methods_022718(prefix,form,n,d,D,Ng,sigma,rho,seed, ...
    initMode,lambda,gamma,tauScheme,lambdaIncr,jobIdx)

rng(seed);
[X, groupsTrue, Omega] = generate_scmd_data_matrix(n,d,D,Ng,sigma,rho,1,seed);
Xnorm = sqrt(sum(X.^2));
X = X ./ repmat(Xnorm, [D 1]);
Xobs = X.*Omega;

if strcmpi(form, 'ENSC_MC_spams')
  solver = ENSC_MC_spams(Xobs, Omega, n, lambda, gamma, 0);
elseif strcmpi(form, 'ENSC_Group_MC_spams')
  solver = ENSC_Group_MC_spams(Xobs, Omega, n, lambda, gamma);
else
  error('formulation not implemented.')
end

opt_params.maxIter = 300; opt_params.convThr = 1e-6; % opt_params.convThr = -Inf;
opt_params.lambdaIncr = lambdaIncr;
opt_params.tauScheme = tauScheme;
opt_params.trueData = {X, groupsTrue};
opt_params.prtLevel = 1; opt_params.logLevel = 3;
exprC_params = struct; compY_params = struct;
% compY_params.maxIter = 2000; compY_params.convThr = 1e-8;
opt_params.maxTime = ceil(60*30 - 60); % ~~30 min.

if strcmpi(initMode, 'pzf_ssc')
  tmp_solver = ENSC_Group_MC_spams(X.*Omega, Omega, n, lambda, 1);
  tmp_opt_params = struct('maxIter', 1, 'tauScheme', [inf 0], ...
      'prtLevel', 0, 'logLevel', 0);
  [~,~,Y0,~] = tmp_solver.solve([], tmp_opt_params);
else
  % default: zero-filling.
  Y0 = [];
end
[groups, C, Y, history] = solver.solve(Y0, opt_params, exprC_params, compY_params);
C = sparse(C);

fname = sprintf('%s_%s_n%d_d%d_D%d_Ng%d_sigma%.0e_rho%.1f_seed%d_init%s_lamb%.1e_gamma%.1f_tauScheme%d-%d_lambIncr%.2f_job%05d.mat', ...
    prefix, form, n, d, D, Ng, sigma, rho, seed, initMode, lambda, gamma, tauScheme, lambdaIncr, jobIdx);

opt_params.trueData = {};
save(fname, 'form', 'n', 'd', 'D', 'Ng', 'sigma', 'rho', 'seed', 'initMode', 'lambda', 'gamma', ...
    'opt_params', 'history', 'jobIdx');
end
