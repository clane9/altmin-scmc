function test_sc_mc(prefix, formulation, n, d, D, Ng, sigma, rho, delta, seed, ...
    form_params, opt_params, exprC_params, compY_params)

if nargin < 12; opt_params = struct; end
if nargin < 13; exprC_params = struct; end
if nargin < 14; compY_params = struct; end

rng(seed);
[X, Omega, groupsTrue] = generate_scmd_data_matrix(n, d, D, Ng, sigma, rho, delta, seed);
Xnorm = sqrt(sum(X.^2));
X = X ./ repmat(Xnorm, [D 1]);

if strcmpi(formulation, 'ENSC_MC')
  solver = ENSC_MC(X.*Omega, Omega, n, form_params.lambda, ...
      form_params.gamma, form_params.eta);
elseif strcmpi(formulation, 'ENSC_Group_MC')
  solver = ENSC_Group_MC(X.*Omega, Omega, n, form_params.lambda,...
      form_params.gamma);
elseif strcmpi(formulation, 'ENSC_MC_apg')
  solver = ENSC_MC_apg(X.*Omega, Omega, n, form_params.lambda, ...
      form_params.gamma);
elseif strcmpi(formulation, 'CASS_MC')
  solver = CASS_MC(X.*Omega, Omega, n, form_params.lambda);
elseif strcmpi(formulation, 'CASS_MC2')
  solver = CASS_MC2(X.*Omega, Omega, n, form_params.lambda);
else
  error('formulation not implemented.')
end

opt_params.trueData = {X, groupsTrue};
opt_params.prtLevel = 2; opt_params.logLevel = 2;
opt_params.maxIter = 5;
opt_params.numThreads = 1;

profile on;
[groups, C, Y, history] = solver.solve(opt_params, exprC_params, compY_params);
profile off;

fname = sprintf('%s_%s_n%d_d%d_D%d_Ng%d_sigma%.0e_rho%.1f_delta%.1f_seed%d.mat', ...
    prefix, formulation, n, d, D, Ng, sigma, rho, delta, seed);
save(fname, 'formulation', 'n', 'd', 'D', 'Ng', 'sigma', 'rho', 'delta', 'seed', ...
    'form_params', 'opt_params', 'exprC_params', 'compY_params', ...
    'groups', 'C', 'Y', 'history');
end
