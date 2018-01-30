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
  solver = ENSC_MC(X.*Omega, Omega, n, form_params.lambda, form_params.gamma);
elseif strcmpi(formulation, 'ENSC_Group_MC')
  solver = ENSC_Group_MC(X.*Omega, Omega, n, form_params.lambda, form_params.gamma);
elseif strcmpi(formulation, 'ENSC_MC2')
  solver = ENSC_MC2(X.*Omega, Omega, n, form_params.lambda, form_params.gamma);
else
  error('formulation not implemented.')
end

opt_params.trueData = {X, groupsTrue};
opt_params.prtLevel = 1; opt_params.logLevel = 2;

[groups, C, Y, history] = solver.solve(opt_params, exprC_params, compY_params);

fname = sprintf('%s_%s_n%d_d%d_D%d_Ng%d_sigma%.0e_rho%.1f_delta%.1f_seed%d.mat', ...
    prefix, formulation, n, d, D, Ng, sigma, rho, delta, seed);
save(fname, 'formulation', 'n', 'd', 'D', 'Ng', 'sigma', 'rho', 'delta', 'seed', ...
    'form_params', 'opt_params', 'exprC_params', 'compY_params', ...
    'groups', 'C', 'Y', 'history');
end
