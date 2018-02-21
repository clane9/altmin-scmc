classdef SDLSC_MC < SC_MC_Base_Solver
% SDLSC_MC    Solver for SDL based combined subspace clustering and completion.
%   Solves formulation
%
%   min_{D,C} lambda/2 ||P_Omega(X - UC)||_F^2 ...
%       + 1/2 ||U||_F^2 + eta_1 ||C||_1 + eta_2/2 ||C||_F^2
%
%   solver = SDLSC_MC(X, Omega, n, lambda, eta1, eta2)

  properties
    eta1; eta2;
  end

  methods

    function self = SDLSC_MC(X, Omega, n, lambda, eta1, eta2)
    self = self@SC_MC_Base_Solver(X, Omega, n, lambda);
    self.eta1 = eta1;
    self.eta2 = eta2;
    end


    function [obj, L, R] = objective(self, U, C)
    Res = self.X - U*C;
    L = 0.5*sum(Res(self.Omega).^2);
    R = 0.5*sum(U(:).^2) + self.eta1*sum(abs(C(:))) + self.eta2*0.5*sum(C(:).^2);
    obj = self.lambda*L + R;
    end


    function [groups, C, Y, history, U] = solve(self, U0, params, exprC_params, ...
        solveU_params)
    % solve   solve alternating minimization for SDLSC_MC joint subspace clustering and
    %   completion.
    %
    %   [groups, C, Y, history, U] = solver.solve(U0, params, exprC_params,
    %       solveU_params)
    %
    %   Args:
    %     U0: Initial guess for dictionary U. If empty, will initialize with
    %        N elements of ZF data.
    %     params: struct containing parameters for optimization:
    %       K: Size of default dictionary [default: 0.2*N].
    %       maxIter: maximum iterations [default: 30].
    %       maxTime: maximum time allowed in seconds [default: Inf].
    %       convThr: convergence threshold [default: 1e-6].
    %       lambdaIncr: rate for adjusting lambda during optimization
    %         [default: 1].
    %       lambdaIncrSteps: How often to increase lambda [default: 50].
    %       trueData: cell containing {Xtrue, groupsTrue} if available
    %         [default: {}].
    %       prtLevel: printing level 0=none, 1=outer iteration, 2=outer &
    %         sub-problem iteration [default: 1].
    %       logLevel: logging level 0=minimal, 1=outer iteration info, 2=outer
    %         & sub-problem iteration info [default: 1].
    %     exprC_params: parameters for self-expression sub-problem [default:
    %       see solver.exprC].
    %     solveU_params: parameters for completion sub-problem [default:
    %       see solver.solveU].
    %
    %   Returns:
    %     groups: N x 1 cluster assignment
    %     C: K x N sparse representation coefficient C.
    %     Y: D x N completed data.
    %     history: optimization log.
    %     U: D x K dictionary.

    % Set defaults.
    if nargin < 2; params = struct; end
    if nargin < 3; exprC_params = struct; end
    if nargin < 4; solveU_params = struct; end
    fields = {'K', 'maxIter', 'maxTime', 'convThr', 'lambdaIncr', ...
        'lambdaIncrSteps', 'trueData', 'prtLevel', 'logLevel'};
    defaults = {ceil(0.2*self.N), 30, Inf, 1e-6, 1, 50, {}, 1, 1};
    for ii=1:length(fields)
      if ~isfield(params, fields{ii})
        params.(fields{ii}) = defaults{ii};
      end
    end
    exprC_params.prtLevel = params.prtLevel-1;
    exprC_params.logLevel = params.logLevel-1;
    solveU_params.prtLevel = params.prtLevel-1;
    solveU_params.logLevel = params.logLevel-1;
    tstart = tic; % start timer.

    % Initialize constants.
    evaltrue = false;
    if ~isempty(params.trueData)
      evaltrue = true;
      [Xtrue, groupsTrue] = params.trueData{:};
    end

    prtformstr = ['(main alt) k=%d, obj=%.2e, ' ...
        'convobj=%.2e, convC=%.2e, convU=%.2e, rtime=%.2f,%.2f'];
    if evaltrue
      prtformstr = [prtformstr ', cmperr=%.3f, clstrerr=%.3f, reconerr=%.3f'];
    end
    prtformstr = [prtformstr ' \n'];

    if isempty(U0)
      subinds = randperm(self.N, params.K);
      U = self.X(:,subinds);
    else
      U = U0;
    end
    K = size(U,2); C = zeros(K, self.N);
    relthr = infnorm(self.X(self.Omega));
    U_last = U; C_last = C; obj_last = self.objective(U, C);
    history.status = 1;
    for kk=1:params.maxIter
      if kk > 1
        % Prune dictionary as in K-SVD, to try to avoid local-minima.
        U = self.pruneU(U,C);
      end
      % Alternate updating C, U.
      [C, exprC_history] = self.exprC(U, C, exprC_params);
      [U, solveU_history] = self.solveU(U, C, solveU_params);

      % Diagnostic measures.
      convC = infnorm(C - C_last)/relthr;
      convU = infnorm(U - U_last)/relthr;
      [obj, L, R] = self.objective(U, C);
      convobj = (obj_last - obj)/obj;
      true_scores = [];
      if evaltrue
        comp_err = self.eval_comp_err(U, C, Xtrue);
        [groups, ~, cluster_err] = self.cluster(C, groupsTrue);
        recon_err = sum(sum((Xtrue - Xtrue*C).^2));
        true_scores = [comp_err cluster_err recon_err];
      end

      % Printing, logging.
      if params.prtLevel > 0
        subprob_rts = [exprC_history.rtime solveU_history.rtime];
        fprintf(prtformstr, [kk obj convobj convC convU subprob_rts true_scores]);
      end
      if params.logLevel > 0
        history.obj(kk,:) = [obj L R];
        history.conv(kk,:) = [convobj convC convU];
        if evaltrue
          history.true_scores(kk,:) = true_scores;
        end
        if params.logLevel > 1
          history.exprC_history{kk} = exprC_history;
          history.solveU_history{kk} = solveU_history;
        end
      end

      % Check stopping cond: objective fails to decrease, or iterates don't change.
      if (max(convC, convU) < params.convThr) % || (convobj < params.convThr)
        history.status = 0;
        break
      end
      if toc(tstart) >= params.maxTime
        fprintf('Timeout!\n');
        break
      end
      if max(convC, convU) > 1e5
        fprintf('Divergence!\n');
        history.status = 2;
        break
      end
      C_last = C; U_last = U; obj_last = obj;
      % Increase lambda
      if mod(kk,params.lambdaIncrSteps) == 0
        self.lambda = min(params.lambdaIncr*self.lambda, 1e4);
      end
    end
    history.iter = kk;
    if ~evaltrue
      groups = self.cluster(C, groupsTrue);
    end
    % Form completion.
    Y = U*C;
    history.rtime = toc(tstart);
    end


    function [C, history] = exprC(self, U, ~, ~)
    tstart = tic;
    K = size(U,2);
    C = zeros(K, self.N);
    % Convert to spams notation.
    spams_param.lambda = self.eta1/self.lambda;
    spams_param.lambda2 = self.eta2/self.lambda;
    spams_param.numThreads = 4;
    for jj=1:self.N
      % solve: lambda/2 || (x_j)_{omega_j} - diag(omega_j)U c_j||_2^2 + ...
      % by dropping unobserved rows of x_j, U.
      omegaj = self.Omega(:,jj);
      xj = self.X(:,jj);
      C(:,jj) = mexLasso(xj(omegaj), U(omegaj,:), spams_param);
    end
    history.iter = 0; history.status = 0; history.rtime = toc(tstart);
    end


    function [U, history] = solveU(self, ~, C, ~)
    tstart = tic;
    K = size(C,1);
    U = zeros(self.D, K);
    CT = C';
    eyeK = eye(K);
    for ii=1:self.D
      % solve: lambda/2 || (x_i)_{omega_i} - u_i C diag(omega_i)||_2^2 + ...
      % by dropping unobserved columns of x_i, C (note x_i, omega_i, u_i row vectors.)
      omegai = self.Omega(ii,:)';
      xi = self.X(ii,:)';
      A = CT(omegai,:);
      U(ii,:) = (self.lambda*(A'*A) + eyeK) \ (self.lambda*(A'*xi(omegai)));
    end
    history.iter = 0; history.status = 0; history.rtime = toc(tstart);
    end


    function U = pruneU(self, U, C, toosmall_thr, toosim_thr)
    if nargin < 4; toosmall_thr = 1e-4; end
    if nargin < 4; toosim_thr = 0.9; end
    % Get indices of dictionary elements not used enough.
    K = size(U,2);
    Unorms = sum(U.^2); Cnorms = sum(C.^2,2)';
    combnorms = 0.5*(Unorms + Cnorms);
    toosmallmask = combnorms < toosmall_thr*max(combnorms);
    % Get indices of elements too similar to one another.
    normU = U./repmat(sqrt(Unorms)+eps, [self.D 1]);
    G = normU'*normU;
    % Set lower triangle (including diagonal) to 0.
    I = repmat((1:K)', [1 K]); J = repmat(1:K, [K 1]); triumask = I<=J;
    G(triumask) = 0;
    dotprods = max(abs(G));
    toosimmask = dotprods > toosim_thr;
    % Replace too small/too similar elements with least-well-represented
    % data points.
    prunemask = max(toosmallmask, toosimmask);
    nreplace = sum(prunemask);
    if nreplace > 0
      Res = sum((self.Omega.*(self.X - U*C)).^2);
      [~, leastrepinds] = sort(Res, 'descend');
      leastrepinds = leastrepinds(1:nreplace);
      U(:,prunemask) = self.X(:,leastrepinds);
    end
    end


    function [groups, A, cluster_err] = cluster(self, C, groupsTrue)
    % cluster   Construct affinity and apply spectral clustering.
    %
    %   [groups, A, cluster_err] = solver.cluster(C, groupsTrue)

    % Normalized-cut spectral clustering for bipartite affinity graph.
    % See: Adler et al., "Linear-time subspace clustering", 2015.
    [K, N] = size(C);
    absC = abs(C);
    % Threshold small values to zero.
    absC(absC < 1e-8) = 0; absC = sparse(absC);

    D1 = spdiags(1./sqrt(sum(absC,2)+eps), 0, K, K);
    D2 = spdiags(1./sqrt(sum(absC)+eps)', 0, N, N);
    [U, ~, V] = svds(D1*absC*D2, self.n);
    Z = [D1*U; D2*V];

    MAXiter = 1000; % Maximum number of iterations for KMeans
    REPlic = 20; % Number of replications for KMeans
    groups = kmeans(Z, self.n, 'maxiter', MAXiter, 'replicates', REPlic);
    % First K elements correspond to dictionary atoms.
    groups = groups((K+1):end);

    % Affinity matrix computed for visualization and post-processing only.
    A = [sparse(K+N, K) [absC; sparse(N, N)]];
    A = A + A';
    if nargin > 2
      [cluster_err, groups] = eval_cluster_error(groups, groupsTrue);
    else
      cluster_err = nan;
    end
    end


    function comp_err = eval_comp_err(self, U, C, Xtrue)
    % eval_comp_err   Evaluate completion error of Y = U*C.
    %
    %   comp_err = solver.eval_comp_err(U, C, Xtrue)
    Xunobs = Xtrue(self.Omegac);
    Y = U*C;
    comp_err = norm(Y(self.Omegac)-Xunobs)/norm(Xunobs);
    end


  end
end
