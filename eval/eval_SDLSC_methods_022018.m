function eval_SDLSC_methods_022018(prefix,form,n,d,D,Ng,sigma,rho,seed, ...
    initMode,lambda,eta1,K,lambdaIncr)

rng(seed);
[X, groupsTrue, Omega] = generate_scmd_data_matrix(n,d,D,Ng,sigma,rho,1,seed);
Xnorm = sqrt(sum(X.^2));
X = X ./ repmat(Xnorm, [D 1]);
Xobs = X.*Omega;
N = size(X,2);

if strcmpi(form, 'SDLSC_MC')
  solver = SDLSC_MC(Xobs, Omega, n, lambda, eta1);
else
  error('formulation not implemented.')
end

opt_params.maxIter = 500; opt_params.convThr = -inf; % opt_params.convThr = 1e-6;
opt_params.lambdaIncr = lambdaIncr;
opt_params.trueData = {X, groupsTrue};
opt_params.prtLevel = 1; opt_params.logLevel = 2;
opt_params.maxTime = ceil(60*5 - 60); % ~~5 min.

if strcmpi(initMode, 'true')
  U0 = X;
elseif strcmpi(initMode, 'lrmc')
  U0 = alm_mc(Xobs, Omega);
elseif strcmpi(initMode, 'rand')
  U0 = randn(D,N);
else
  % default: zero-filling.
  U0 = Xobs;
end
if K > 0 && K < N
  subinds = randperm(N,K);
  U0 = U0(:,subinds);
end

[groups, C, Y, history, U] = solver.solve(U0, opt_params);
C = sparse(C);

fname = sprintf('%s_%s_n%d_d%d_D%d_Ng%d_sigma%.0e_rho%.1f_seed%d_init%s_lamb%.1e_eta1%.1e_K%d_lambIncr%.2f.mat', ...
    prefix, form, n, d, D, Ng, sigma, rho, seed, initMode, lambda, eta1, K, lambdaIncr);

opt_params.trueData = {};
save(fname, 'form', 'n', 'd', 'D', 'Ng', 'sigma', 'rho', 'seed', 'initMode', 'lambda', 'eta1', 'K', ...
    'opt_params', 'groups', 'C', 'Y', 'history', 'U');
end
