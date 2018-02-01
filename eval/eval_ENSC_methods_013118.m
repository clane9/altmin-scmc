function eval_ENSC_methods_013118(prefix,form,n,d,D,Ng,rho,seed, ...
    lambda,gamma,tauScheme)

rng(seed);
[X, Omega, groupsTrue] = generate_scmd_data_matrix(n,d,D,Ng,0,rho,1,seed);
Xnorm = sqrt(sum(X.^2));
X = X ./ repmat(Xnorm, [D 1]);
Xobs = X.*Omega;

if strcmpi(form, 'ENSC_MC')
  solver = ENSC_MC(Xobs, Omega, n, lambda, gamma, 0);
elseif strcmpi(form, 'ENSC_MC_apg')
  solver = ENSC_MC_apg(Xobs, Omega, n, lambda, gamma, 0);
elseif strcmpi(form, 'ENSC_Group_MC')
  solver = ENSC_Group_MC(Xobs, Omega, n, lambda, gamma, 0);
else
  error('formulation not implemented.')
end

opt_params.tauScheme = tauScheme;
opt_params.trueData = {X, groupsTrue};
opt_params.prtLevel = 1; opt_params.logLevel = 2;
opt_params.maxIter = 30; opt_params.convThr = 1e-6;

[groups, C, Y, history] = solver.solve(opt_params);

fname = sprintf('%s_%s_n%d_d%d_D%d_Ng%d_rho%.1f_seed%d_lamb%.2e_gamma%.1f_tauScheme%d-%d.mat', ...
    prefix, form, n, d, D, Ng, rho, seed, lambda, gamma, tauScheme);

save(fname, 'form', 'n', 'd', 'D', 'Ng', 'rho', 'seed', 'lambda', 'gamma', 'tauScheme', ...
    'opt_params', 'groups', 'C', 'Y', 'history');
end
