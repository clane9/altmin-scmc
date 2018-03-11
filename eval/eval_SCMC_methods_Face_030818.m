function eval_SCMC_methods_Face_030818(prefix,form,denoised,n,rhos,seed,initMode,...
      lambda,gamma,eta1,Ud,Gamma,alpha,tau1,lambdaIncr,maxIter,maxSecs,jobIdx)

rng(seed);

% Load Yale Face B dataset.
datapath = '~/Documents/Datasets/CroppedYale/small_YaleB_24x21_rpca_denoise.mat';
load(datapath, 'small_yale');
groupsTrue = small_yale.labels(1,:)';
if denoised
  X = small_yale.A; % Result of rpca applied to each subject separately.
else
  X = small_yale.images;
end

% Subset groups if needed.
if n < max(groupsTrue)
  subclasses = sort(randperm(max(groupsTrue), n));
  % subclasses = 1:n;
  [subXs, subgroups] = deal(cell(1,n));
  for ii=1:n
    Ind = groupsTrue == subclasses(ii);
    subXs{ii} = X(:,Ind);
    subgroups{ii} = ii*ones(1,size(subXs{ii},2));
  end
  X = [subXs{:}]; groupsTrue = [subgroups{:}]';
else
  n = max(groupsTrue);
  subclasses = 1:n;
end
[D, N] = size(X);

% Normalize.
Xnorm = sqrt(sum(X.^2))+eps;
X = X ./ repmat(Xnorm, [D 1]);

% Initialize results table
nrho = length(rhos);
tablefields = {'rho', 'comperr', 'clustererr'};
tabledata = cell(nrho, length(tablefields));

% Divide total time by number of trials.
% Not the best since some trials may take longer than others.
maxSecs = maxSecs / nrho;

% Parameters for initialization.
init_params.lambda = 320;
init_params.tauScheme = [inf 0];
init_params.lambdaIncr = 1.0;
init_params.maxIter = 10;

% Optimization parameters for alternating min methods.
opt_params.maxIter = maxIter;
opt_params.convThr = 1e-6;
opt_params.lambdaIncr = lambdaIncr;
opt_params.tauScheme = [tau1, 0];
opt_params.trueData = {X, groupsTrue}; % true groups should be column vector.
opt_params.prtLevel = 1; opt_params.logLevel = 1;

% Parameters for S3LR (copied from defaults).
gamma0 = Gamma;
relax = 1; affine = 0;
opt.tol =1e-4;
opt.maxIter = 500; % maximum subproblem iters comparable to other altmin methods.
opt.rho =1.1;
opt.mu_max =1e4;
opt.norm_sr ='1';
opt.norm_mc ='1';
T = 1;

% Store histories for those methods that return one.
histories = cell(1,nrho);

for jj=1:nrho
  % Start timer for purpose of controlling trial runtime.
  tstart = tic;

  % Sample pattern of missing entries.
  M = ceil(rhos(jj)*D*N);
  Omega = true(D, N); Omega(randperm(D*N, M)) = false;
  Xobs = X.*Omega; Xunobs = X(~Omega);

  % Initialization.
  if strcmpi(initMode, 'lrmc')
    Y0 = alm_mc(Xobs, Omega);
  elseif strcmpi(initMode, 'pzf_ssc')
    tmp_solver = ENSC_Group_MC_spams(Xobs, Omega, n, init_params.lambda, 1);
    tmp_opt_params = struct('maxIter', 1, 'tauScheme', init_params.tauScheme, ...
        'prtLevel', 0, 'logLevel', 0);
    [tmp_groups,~,Y0,~] = tmp_solver.solve([], tmp_opt_params);
  elseif strcmpi(initMode, 'alt_pzf_ssc')
    tmp_solver = ENSC_Group_MC_spams(Xobs, Omega, n, init_params.lambda, 1);
    tmp_opt_params = struct('maxIter', init_params.maxIter, 'tauScheme', init_params.tauScheme, ...
        'lambdaIncr', init_params.lambdaIncr, 'prtLevel', 0, 'logLevel', 0);
    [tmp_groups,~,Y0,~] = tmp_solver.solve([], tmp_opt_params);
  else
    % Default: zero-filling.
    Y0 = []; tmp_groups = [];
  end

  % Update runtime limit based on how long initialization took.
  opt_params.maxTime = maxSecs - toc(tstart);

  % Run clustering and completion.
  if strcmpi(form, 'LRMC_ENSC')
    solver = ENSC_Group_MC_spams(Xobs, Omega, n, lambda, gamma);
    opt_params.maxIter = 1;
    opt_params.lambdaIncr = 1.0;
    opt_params.tauScheme = [0, 0]; % No projection.
    Y = alm_mc(Xobs, Omega);
    [~, C, ~, histories{jj}] = solver.solve(Y, opt_params);
  elseif strcmpi(form, 'PZF_ENSC_LRMC')
    solver = ENSC_Group_MC_spams(Xobs, Omega, n, lambda, gamma);
    opt_params.maxIter = 1;
    opt_params.lambdaIncr = 1.0;
    opt_params.tauScheme = [Inf, 0];
    [~, C, Y, histories{jj}] = solver.solve(Y0, opt_params);
  elseif strcmpi(form, 'Alt_PZF_ENSC_LRMC')
    solver = ENSC_Group_MC_spams(Xobs, Omega, n, lambda, gamma);
    [~, C, Y, histories{jj}] = solver.solve(Y0, opt_params);
  elseif strcmpi(form, 'ENSC_MC')
    solver = ENSC_MC_spams_comp_apg(Xobs, Omega, n, lambda, gamma);
    [~, C, Y, histories{jj}] = solver.solve(Y0, opt_params);
  elseif strcmpi(form, 'GSSC_MC')
    solver = GSSC_MC2(Xobs, Omega, n, lambda, eta1, Ud);
    % For dictionary learning methods, initialize by group-wise SVD.
    if ~isempty(tmp_groups)
      U0 = zeros(D, Ud*n);
      for kk=1:n
        [Uk, ~, ~] = svd(Y0(:,tmp_groups==kk));
        Uk = Uk(:,1:Ud);
        startind = (kk-1)*Ud + 1; stopind = kk*Ud;
        U0(:,startind:stopind) = Uk;
      end
    else
      U0 = [];
    end
    opt_params.Y0 = Y0;
    [~, C, Y, histories{jj}] = solver.solve(U0, opt_params);
  elseif strcmpi(form, 'S3LR')
    [~, ~, C, Y, ~] = S3LR(Xobs, Omega, groupsTrue, X, lambda, Gamma, Y0, gamma0, ...
        relax, affine, maxIter, maxSecs - toc(tstart), opt, T, lambdaIncr);
  elseif strcmpi(form, 'SRME_MC')
    opt_params.maxIter = opt_params.maxIter*10; % Increase iterations since no subproblem.
    [Y, C, ~] = SRME_MC_ADMM(Xobs, Omega, Y0, lambda, alpha, opt_params);
  else
    error('formulation not implemented!')
  end

  % Evaluate clustering error and completion.
  % GSSC clusters in a different way than other methods.
  if strcmpi(form, 'GSSC_MC')
    [~, ~, cluster_err] = solver.cluster(C, groupsTrue);
  else
    A = build_affinity(C);
    groups = spectral_clustering(A, n);
    [cluster_err, ~] = eval_cluster_error(groups, groupsTrue);
  end
  comp_err = norm(Y(~Omega) - Xunobs)/norm(Xunobs);

  fprintf('\nk=%d, rho=%.2f, comperr=%.3f, clustererr=%.3f \n\n', ...
      jj, rhos(jj), comp_err, cluster_err);

  % Insert results into table.
  tabledata(jj, :) = {rhos(jj), comp_err, cluster_err};
end

fname = sprintf('%s_%s_rpca%d_n%d_seed%d_init%s_lamb%.1e_gamma%.1f_eta1%.1e_Ud%d_Gamma%.1e_alpha%.1e_tau1%d_lambIncr%.2f_job%05d.mat', ...
    prefix, form, denoised, n, seed, initMode, lambda, gamma, eta1, Ud, Gamma, alpha, ...
    tau1, lambdaIncr, jobIdx);

save(fname, 'form', 'denoised', 'n', 'subclasses', 'rhos', 'seed', 'initMode', 'lambda', 'gamma', 'eta1', 'Ud', 'Gamma', 'alpha', ...
    'tau1', 'lambdaIncr', 'tablefields', 'tabledata', 'histories', 'jobIdx');

end
