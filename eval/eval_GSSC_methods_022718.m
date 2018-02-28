function eval_GSSC_methods_022718(prefix,n,d,D,Ng,sigma,rho,seed, ...
    initMode,lambda,eta1,Ud,lambdaIncr,jobIdx)

rng(seed);
[X, groupsTrue, Omega] = generate_scmd_data_matrix(n,d,D,Ng,sigma,rho,1,seed);
Xnorm = sqrt(sum(X.^2));
X = X ./ repmat(Xnorm, [D 1]);
Xobs = X.*Omega;

form = 'GSSC_MC';
solver = GSSC_MC2(Xobs, Omega, n, lambda, eta1, Ud);

opt_params.maxIter = 300; opt_params.convThr = 1e-6; % opt_params.convThr = -inf;
opt_params.lambdaIncr = lambdaIncr;
opt_params.trueData = {X, groupsTrue};
opt_params.prtLevel = 1; opt_params.logLevel = 2;
opt_params.maxTime = ceil(60*20 - 60); % ~~20 min.

if strcmpi(initMode, 'pzf_ssc')
  tmp_solver = ENSC_Group_MC_spams(X.*Omega, Omega, n, lambda, 1);
  tmp_opt_params = struct('maxIter', 1, 'tauScheme', [inf 0], ...
      'prtLevel', 0, 'logLevel', 0);
  [tmp_groups,~,Y0,~] = tmp_solver.solve([], tmp_opt_params);
  opt_params.Y0 = Y0;
  U0 = zeros(D, Ud*n);
  for ii=1:n
    [Ui, Si, ~] = svds(Y0(:,tmp_groups==ii),Ud);
    startind = (ii-1)*Ud + 1; stopind = ii*Ud;
    U0(:,startind:stopind) = Ui; % Ui*Si;
  end
else
  % default: zero-filling.
  U0 = [];
end

[groups, C, Y, history, U] = solver.solve(U0, opt_params);
C = sparse(C);

fname = sprintf('%s_%s_n%d_d%d_D%d_Ng%d_sigma%.0e_rho%.1f_seed%d_init%s_lamb%.1e_eta1%.1e_Ud%d_lambIncr%.2f_job%05d.mat', ...
    prefix, form, n, d, D, Ng, sigma, rho, seed, initMode, lambda, eta1, Ud, lambdaIncr, jobIdx);

opt_params.trueData = {}; opt_params.Y0 = [];
save(fname, 'form', 'n', 'd', 'D', 'Ng', 'sigma', 'rho', 'seed', 'initMode', 'lambda', 'eta1', 'Ud', ...
    'opt_params', 'history', 'jobIdx');
end
