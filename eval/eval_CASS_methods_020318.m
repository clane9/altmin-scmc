function eval_CASS_methods_020318(prefix,form,n,d,D,Ng,rho,seed, ...
    lambda,tauScheme)

rng(seed);
[X, groupsTrue, Omega] = generate_scmd_data_matrix(n,d,D,Ng,0,rho,1,seed);
Xnorm = sqrt(sum(X.^2));
X = X ./ repmat(Xnorm, [D 1]);
Xobs = X.*Omega;

if strcmpi(form, 'CASS_MC')
  solver = CASS_MC(Xobs, Omega, n, lambda);
elseif strcmpi(form, 'CASS_MC2')
  solver = CASS_MC2(Xobs, Omega, n, lambda);
else
  error('formulation not implemented.')
end

opt_params.tauScheme = tauScheme;
opt_params.trueData = {X, groupsTrue};
opt_params.prtLevel = 1; opt_params.logLevel = 2;
opt_params.maxIter = 20; opt_params.convThr = 1e-3; % Relaxed these relative to previous.
opt_params.maxTime = ceil(60^2 - 60); % ~~1 hr.

[groups, C, Y, history] = solver.solve(opt_params);

fname = sprintf('%s_%s_n%d_d%d_D%d_Ng%d_rho%.1f_seed%d_lamb%.2e_tauScheme%d-%d.mat', ...
    prefix, form, n, d, D, Ng, rho, seed, lambda, tauScheme);

save(fname, 'form', 'n', 'd', 'D', 'Ng', 'rho', 'seed', 'lambda', 'tauScheme', ...
    'opt_params', 'groups', 'C', 'Y', 'history');
end
