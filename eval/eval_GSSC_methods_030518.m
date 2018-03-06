function eval_GSSC_methods_030518(prefix,n,d,D,Ng,sigma,rho,seed, ...
    initMode,lambda,eta1,Ud,lambdaIncr,maxIter,maxSecs,jobIdx)

rng(seed);
[X, groupsTrue, Omega] = generate_scmd_data_matrix(n,d,D,Ng,sigma,rho,1,seed);
Xnorm = sqrt(sum(X.^2));
X = X ./ repmat(Xnorm, [D 1]);
Xobs = X.*Omega;

form = 'GSSC_MC';
solver = GSSC_MC2(Xobs, Omega, n, lambda, eta1, Ud);

opt_params.maxIter = maxIter; opt_params.convThr = 1e-6; % opt_params.convThr = -inf;
opt_params.lambdaIncr = lambdaIncr;
opt_params.trueData = {X, groupsTrue};
opt_params.prtLevel = 1; opt_params.logLevel = 2;

tic;
init_params.lambda = 20;
init_params.tauScheme = [inf 0];
init_params.lambdaIncr = 1.1;
init_params.maxIter = 10;
if any(strcmpi(initMode, {'pzf_ssc', 'alt_pzf_ssc'}))
  if strcmpi(initMode, 'pzf_ssc')
    tmp_solver = ENSC_Group_MC_spams(Xobs, Omega, n, init_params.lambda, 1);
    tmp_opt_params = struct('maxIter', 1, 'tauScheme', init_params.tauScheme, ...
        'prtLevel', 0, 'logLevel', 0);
    [tmp_groups,~,Y0,~] = tmp_solver.solve([], tmp_opt_params);
  else
    tmp_solver = ENSC_Group_MC_spams(Xobs, Omega, n, init_params.lambda, 1);
    tmp_opt_params = struct('maxIter', init_params.maxIter, 'tauScheme', init_params.tauScheme, ...
        'lambdaIncr', init_params.lambdaIncr, 'prtLevel', 0, 'logLevel', 0);
    [tmp_groups,~,Y0,~] = tmp_solver.solve([], tmp_opt_params);
  end
  opt_params.Y0 = Y0;
  U0 = zeros(D, Ud*n);
  for ii=1:n
    [Ui, Si, ~] = svd(Y0(:,tmp_groups==ii));
    Ui = Ui(:,1:Ud);
    startind = (ii-1)*Ud + 1; stopind = ii*Ud;
    U0(:,startind:stopind) = Ui; % Ui*Si;
  end
else
  % default: zero-filling.
  U0 = [];
end
opt_params.maxTime = maxSecs - toc;

[~, ~, ~, history, ~] = solver.solve(U0, opt_params);

fname = sprintf('%s_%s_n%d_d%d_D%d_Ng%d_sigma%.0e_rho%.2f_seed%d_init%s_lamb%.1e_eta1%.1e_Ud%d_lambIncr%.2f_job%05d.mat', ...
    prefix, form, n, d, D, Ng, sigma, rho, seed, initMode, lambda, eta1, Ud, lambdaIncr, jobIdx);

opt_params.trueData = {}; opt_params.Y0 = [];
save(fname, 'form', 'n', 'd', 'D', 'Ng', 'sigma', 'rho', 'seed', 'initMode', 'lambda', 'eta1', 'Ud', ...
    'opt_params', 'history', 'jobIdx');
end
