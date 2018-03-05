function test_sc_mc(prefix, formulation, n, d, D, Ng, sigma, rho, delta, seed, initmode, ...
    form_params, opt_params, exprC_params, compY_params)

if nargin < 12; opt_params = struct; end
if nargin < 13; exprC_params = struct; end
if nargin < 14; compY_params = struct; end

compY_params.maxIter = 200; compY_params.convThr = 1e-3;
exprC_params.maxIter = 200; exprC_params.convThr = 1e-3;
% exprC_params.mu = form_params.lambda;

rng(seed);
[X, groupsTrue, Omega] = generate_scmd_data_matrix(n, d, D, Ng, sigma, rho, delta, seed);
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
      form_params.gamma, form_params.eta);
elseif strcmpi(formulation, 'ENSC_MC_spams')
  solver = ENSC_MC_spams(X.*Omega, Omega, n, form_params.lambda, ...
      form_params.gamma, form_params.eta);
elseif strcmpi(formulation, 'ENSC_Group_MC_spams')
  solver = ENSC_Group_MC_spams(X.*Omega, Omega, n, form_params.lambda,...
      form_params.gamma);
elseif strcmpi(formulation, 'CASS_MC')
  solver = CASS_MC(X.*Omega, Omega, n, form_params.lambda);
elseif strcmpi(formulation, 'ENSC_CASS_MC')
  solver = ENSC_CASS_MC(X.*Omega, Omega, n, form_params.lambda, ...
      form_params.gamma);
elseif strcmpi(formulation, 'LRSC_MC')
  solver = LRSC_MC(X.*Omega, Omega, n, form_params.lambda, ...
      form_params.eta);
elseif strcmpi(formulation, 'LSRSC_MC')
  solver = LSRSC_MC(X.*Omega, Omega, n, form_params.lambda, ...
      form_params.eta);
elseif strcmpi(formulation, 'SDLSC_MC')
  solver = SDLSC_MC(X.*Omega, Omega, n, form_params.lambda, ...
      form_params.eta1, form_params.eta2, form_params.K);
elseif strcmpi(formulation, 'SDLSC_MC2')
  solver = SDLSC_MC2(X.*Omega, Omega, n, form_params.lambda, ...
      form_params.eta1, form_params.K);
elseif strcmpi(formulation, 'GSSC_MC')
  solver = GSSC_MC(X.*Omega, Omega, n, form_params.lambda, ...
      form_params.Uconstrain, form_params.eta1, form_params.d);
elseif strcmpi(formulation, 'GSSC_MC2')
  solver = GSSC_MC2(X.*Omega, Omega, n, form_params.lambda, ...
      form_params.eta1, form_params.d);
elseif strcmpi(formulation, 'GSSC_MC_l0')
  solver = GSSC_MC_l0(X.*Omega, Omega, n, form_params.lambda, ...
      form_params.Uconstrain, form_params.eta1, form_params.d);
else
  error('formulation not implemented.')
end

opt_params.trueData = {X, groupsTrue};
opt_params.prtLevel = 1; opt_params.logLevel = 2;
opt_params.maxIter = 50;

if strcmpi(initmode, 'zf')
  Y0 = X.*Omega;
elseif strcmpi(initmode, 'rand')
  Y0 = randn(size(X));
elseif strcmpi(initmode, 'lrmc')
  Y0 = alm_mc(X.*Omega, Omega);
elseif strcmpi(initmode, 'pzf_ssc')
  tmp_solver = ENSC_Group_MC_spams(X.*Omega, Omega, n, 20, 1);
  tmp_opt_params = struct('maxIter', 1, 'tauScheme', [inf 0], ...
      'prtLevel', 0, 'logLevel', 0);
  [tmp_groups,~,Y0,~] = tmp_solver.solve([], tmp_opt_params);
elseif strcmpi(initmode, 'alt_pzf_ssc')
  tmp_solver = ENSC_Group_MC_spams(X.*Omega, Omega, n, 20, 1);
  tmp_opt_params = struct('maxIter', 10, 'tauScheme', [inf 0], ...
      'prtLevel', 0, 'logLevel', 0);
  [tmp_groups,~,Y0,~] = tmp_solver.solve([], tmp_opt_params);
elseif strcmpi(initmode, 'true')
  Y0 = X;
end

% Choose random subset of columns if using SDL method
if any(strcmpi(formulation, {'SDLSC_MC', 'SDLSC_MC2', 'GSSC_MC', 'GSSC_MC_l0', 'GSSC_MC2'}))
  if any(strcmpi(initmode, {'pzf_ssc', 'alt_pzf_ssc'}))
    U0 = zeros(D, solver.K); d = solver.K/n;
    for ii=1:n
      [Ui, Si, ~] = svds(Y0(:,tmp_groups==ii),d);
      startind = (ii-1)*d + 1; stopind = ii*d;
      U0(:,startind:stopind) = Ui; % Ui*Si;
    end
  else
    subinds = randperm(Ng*n, solver.K);
    U0 = Y0(:,subinds);
  end
  Y0 = U0;
end

profile on;
[groups, C, Y, history] = solver.solve(Y0, opt_params, exprC_params, compY_params);
profile off;

fname = sprintf('%s_%s_n%d_d%d_D%d_Ng%d_sigma%.0e_rho%.1f_delta%.1f_seed%d.mat', ...
    prefix, formulation, n, d, D, Ng, sigma, rho, delta, seed);
save(fname, 'formulation', 'n', 'd', 'D', 'Ng', 'sigma', 'rho', 'delta', 'seed', ...
    'form_params', 'opt_params', 'exprC_params', 'compY_params', 'initmode', ...
    'groups', 'C', 'Y', 'history');
end
