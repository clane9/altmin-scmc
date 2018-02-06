function eval_SCMC_methods_Img_020518(prefix,dataset,form,n,D,Ng,rho,seed,...
    lambda,gamma,tauScheme,maxIter,maxTime)

if nargin < 10; tauScheme = [Inf 0]; end
if nargin < 11; maxIter = 20; end
if nargin < 11; maxTime = 60^2; end

rng(seed);

tstart = tic;
% Load dataset.
if strcmpi(dataset, 'COIL20_SC')
  load COIL20_SC.mat COIL20_SC_DATA COIL20_LABEL;
  X = COIL20_SC_DATA; groupsTrue = COIL20_LABEL;
  clear COIL20_SC_DATA COIL20_LABEL;
elseif strcmpi(dataset, 'MNIST_SC')
  load MNIST_SC.mat MNIST_SC_DATA MNIST_LABEL;
  X = MNIST_SC_DATA; groupsTrue = MNIST_LABEL;
  clear MNIST_SC_DATA MNIST_LABEL;
else
  error('dataset not supported!')
end

% Downsample and select subset of data-points.
X = dimReduction_PCA(X, D);
labels = unique(groupsTrue);
sublabels = labels(randperm(length(labels), n));
samp_mask = false(size(groupsTrue));
for ii=1:n
  lab_idx = find(groupsTrue==sublabels(ii));
  lab_n = length(lab_idx);
  lab_samp = lab_idx(randperm(lab_n, min([Ng lab_n])));
  samp_mask(lab_samp) = true;
end
X = X(:,samp_mask); groupsTrue = groupsTrue(samp_mask);
Xnorm = sqrt(sum(X.^2));
X = X ./ repmat(Xnorm, [D 1]);
N = size(X,2);

% Construct pattern of missing entries.
M = round(rho*D*N);
unobsInd = randperm(D*N, M);
Omega = true(D, N); Omega(unobsInd) = false;
Xobs = X.*Omega;

% Run SC+MC.
if any(strcmpi(form, {'CASS_MC2', 'ENSC_MC', 'ENSC_Group_MC'}))
  if strcmpi(form, 'CASS_MC2')
    solver = CASS_MC2(Xobs, Omega, n, lambda);
  elseif strcmpi(form, 'ENSC_MC')
    solver = ENSC_MC(Xobs, Omega, n, lambda, gamma, 0);
  elseif strcmpi(form, 'ENSC_Group_MC')
    solver = ENSC_Group_MC(Xobs, Omega, n, lambda, gamma);
  else
    error('formulation not implemented.')
  end
  opt_params.tauScheme = tauScheme;
  opt_params.trueData = {X, groupsTrue};
  opt_params.prtLevel = 1; opt_params.logLevel = 2;
  opt_params.convThr = 1e-3;
  opt_params.maxIter = maxIter; 
  opt_params.maxTime = maxTime - toc(tstart) - ...
      (1/maxIter)*maxTime;
  [groups, C, Y, history] = solver.solve(opt_params);
elseif strcmpi(form, 'SRME_MC')
  opt_params = struct; % not used.
  [groups, C, Y, history] = SRME_MC_ADMM_ron(Xobs,Omega,n,lambda,gamma);
  Xunobs = X(~Omega);
  comp_err = norm(Y(~Omega) - Xunobs)/norm(Xunobs);
  [cluster_err, groups] = eval_cluster_error(groups, groupsTrue);
  history.true_scores = [comp_err cluster_err];
elseif strcmpi(form, 'ENSC')
  opt_params.Nsample = 0.5*Ng;
  [cluster_err, groups, ~, ~, ~, C] = myEnSC(X, groupsTrue, lambda, gamma, ...
      opt_params.Nsample);
  Y = []; history.true_scores = [cluster_err 0];
else
  error('formulation not supported!');
end
C = sparse(C);

% Save
fname = sprintf('%s_%s_%s_n%d_D%d_Ng%d_rho%.1f_seed%d_lamb%.2e_gamma%.2e_tauScheme%d-%d.mat', ...
    prefix, dataset, form, n, D, Ng, rho, seed, lambda, gamma, tauScheme);
save(fname, 'dataset', 'form', 'n', 'D', 'Ng', 'rho', 'seed', 'lambda', 'gamma', ...
    'tauScheme', 'maxIter', 'maxTime', ...
    'opt_params', 'groups', 'C', 'Y', 'history');
end
