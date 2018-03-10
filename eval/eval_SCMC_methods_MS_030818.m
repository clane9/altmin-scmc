function eval_SCMC_methods_MS_030818(prefix,form,rhos,seed,initMode,...
      lambda,gamma,eta1,Ud,Gamma,alpha,tau1,lambdaIncr,maxIter,maxSecs,jobIdx)

rng(seed);

% Load Hopkins 155 motion segmentation dataset.
datapath = '~/Documents/Datasets/Hopkins155';
Trajs = load_Hopkins155(datapath);

% Initialize results table
ntraj = length(Trajs);
if ntraj ~= 156
  error('Only %d trajectories found, expecting 156!', ntraj);
end
nrho = length(rhos);
ntrial = ntraj*nrho;
tablefields = {'trajidx', 'trajname', 'n', 'rho', 'comperr', 'clustererr'};
tabledata = cell(ntrial, length(tablefields));

% Divide total time by number of trials.
% Not the best since some trials will take longer than others.
maxSecs = maxSecs / ntrial;

% Parameters for initialization.
init_params.lambda = 80;
init_params.tauScheme = [inf 0];
init_params.lambdaIncr = 1.0;
init_params.maxIter = 10;

% Optimization parameters for alternating min methods.
opt_params.maxIter = maxIter;
opt_params.convThr = 1e-6;
opt_params.lambdaIncr = lambdaIncr;
opt_params.tauScheme = [tau1, 0];
opt_params.trueData = {};
opt_params.prtLevel = 1; opt_params.logLevel = 0;

% Parameters for S3LR (copied from defaults).
gamma0 = Gamma;
relax = 1; affine = 0;
opt.tol =1e-4;
opt.maxIter =1e6;
opt.rho =1.1;
opt.mu_max =1e4;
opt.norm_sr ='1';
opt.norm_mc ='1';
T = 1;

trialidx = 1;
for ii=1:ntraj
  X = Trajs{ii}.X;
  [D, N] = size(X);
  groupsTrue = Trajs{ii}.s;
  n = Trajs{ii}.n;

  % Sort for visualizing convenience.
  [~, Ind] = sort(groupsTrue);
  X = X(:, Ind); groupsTrue = groupsTrue(Ind);

  % Normalize.
  Xnorm = sqrt(sum(X.^2))+eps;
  X = X ./ repmat(Xnorm, [D 1]);

  opt_params.trueData = {X, groupsTrue};

  for jj=1:nrho
    % Start timer for purpose of controlling trial runtime.
    tstart = tic;

    % Sample pattern of missing entries.
    M = ceil(rhos(jj)*D*N);
    Omega = true(D, N); Omega(randperm(D*N, M)) = false;
    Xobs = X.*Omega; Xunobs = X(~Omega);

    % Initialization.
    if strcmpi(initMode, 'pzf_ssc')
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
    if strcmpi(form, 'ENSC_MC')
      solver = ENSC_MC_spams_comp_apg(Xobs, Omega, n, lambda, gamma);
      [~, C, Y, ~] = solver.solve(Y0, opt_params);
    elseif strcmpi(form, 'ENSC_Group_MC')
      solver = ENSC_Group_MC_spams(Xobs, Omega, n, lambda, gamma);
      [~, C, Y, ~] = solver.solve(Y0, opt_params);
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
      [~, C, Y, ~] = solver.solve(U0, opt_params);
    elseif strcmpi(form, 'S3LR')
      [~, ~, C, Y, ~] = S3LR(Xobs, Omega, groupsTrue, X, lambda, Gamma, Y0, gamma0, ...
          relax, affine, maxIter, maxSecs - toc(tstart), opt, T, lambdaIncr);
    elseif strcmpi(form, 'SRME_MC')
      [Y, C, ~] = SRME_MC_ADMM(Xobs, Omega, Y0, lambda, alpha, opt_params);
    else
      error('formulation not implemented!')
    end

    % Evaluate clustering error and completion.
    A = build_affinity(C);
    groups = spectral_clustering(A, n);
    [cluster_err, ~] = eval_cluster_error(groups, groupsTrue);
    comp_err = norm(Y(~Omega) - Xunobs)/norm(Xunobs);

    fprintf('\nk=%d, traj=%s, n=%d, rho=%.2f, comperr=%.3f, clustererr=%.3f \n\n', ...
        ii, Trajs{ii}.name, rhos(jj), comp_err, cluster_err);

    % Insert results into table.
    tabledata(trialidx, :) = {ii, Trajs{ii}.name, n, rhos(jj), comp_err, cluster_err};
    trialidx = trialidx + 1;
  end
end

fname = sprintf('%s_%s_seed%d_init%s_lamb%.1e_gamma%.1f_eta1%.1e_Ud%d_Gamma%.1e_alpha%.1e_tau1%d_lambIncr%.2f_job%05d.mat', ...
    prefix, form, seed, initMode, lambda, gamma, eta1, Ud, Gamma, alpha, tau1, lambdaIncr, jobIdx);

save(fname, 'form', 'rhos', 'seed', 'initMode', 'lambda', 'gamma', 'eta1', 'Ud', 'Gamma', 'alpha', ...
    'tau1', 'lambdaIncr', 'tablefields', 'tabledata', 'jobidx');

end
