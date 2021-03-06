classdef SC_MC_Base_Solver
% SC_MC_Base_Solver    Base class for alternating minimization algorithms for
%   joint subspace clustering and completion.
%
%   solver = SC_MC_Base_Solver(X, Omega, n, lambda)

  properties
    X; D; N; Omega; Omegac; n; lambda;
  end

  methods

    function self = SC_MC_Base_Solver(X, Omega, n, lambda)
    % SC_MC_Base_Solver    Base class for alternating minimization algorithms for
    %   joint subspace clustering and completion.
    %
    %   solver = SC_MC_Base_Solver(X, Omega, n, lambda)
    %
    %   Args:
    %     X: D x N incomplete data matrix.
    %     Omega: D x N binary pattern of missing entries.
    %     n: number of clusters.
    %     lambda: self-expression penalty parameter.
    %
    %   Returns:
    %     self: SC_MC_Base_Solver instance.
    self.X = X; self.Omega = logical(Omega);
    [self.D, self.N] = size(X);
    self.n = n; self.lambda = lambda;
    self.Omegac = ~self.Omega;
    self.X(self.Omegac) = 0; % zero-fill missing entries.
    end

    function [groups, C, Y, history] = solve(self, Y0, params, exprC_params, ...
        compY_params)
    % solve   solve alternating minimization for joint subspace clustering and
    %   completion. Minimizes a generic objective
    %
    %   min_{Y,C} lambda/2 ||W_k \odot (Y - YC)||_F^2 + g(Y,C)
    %   s.t. diag(C) = 0
    %
    %   [groups, C, Y, history] = solver.solve(Y0, params, exprC_params, compY_params)
    %
    %   Args:
    %     Y0: Initial guess for completion Y. If empty, will initialize with ZF
    %       data.
    %     params: struct containing parameters for optimization:
    %       maxIter: maximum iterations [default: 30].
    %       maxTime: maximum time allowed in seconds [default: Inf].
    %       convThr: convergence threshold [default: 1e-6].
    %       tauScheme: scheme for how the weight tau on unobserved entries
    %         should be set. Must be pair of numbers [t_1 t_2] in [0, inf].
    %         Then we set [tau_k,1 tau_k,2] = ((k-1)/maxIter).^[t_1 t_2], where
    %         tau_k,1 constrols self-expression, and tau_k,2 completion. E.g.
    %         tauScheme = [0 0] sets tau=1 for all iterations, while
    %         tauScheme=[inf inf] sets tau=0 [default: [inf inf]].
    %       lambdaIncr: rate for adjusting lambda during optimization
    %         [default: 1].
    %       lambdaIncrSteps: How often to increase lambda [default: 1].
    %       lambdaMax: Maximum lambda can increase to [default: 1e4].
    %       valData: Data to use for validation, a cell {val_inds, Xval}
    %         corresponding to a set of held-out indices and observed values.
    %         val_inds should be a subset of the indices in Omega^c [default: {}].
    %       trueData: cell containing {Xtrue, groupsTrue} if available
    %         [default: {}].
    %       prtLevel: printing level 0=none, 1=outer iteration, 2=outer &
    %         sub-problem iteration [default: 1].
    %       logLevel: logging level 0=minimal, 1=outer iteration info, 2=outer
    %         & sub-problem iteration info [default: 1].
    %     exprC_params: parameters for self-expression sub-problem [default:
    %       see solver.exprC].
    %     compY_params: parameters for completion sub-problem [default:
    %       see solver.compY].
    %
    %   Returns:
    %     groups: N x 1 cluster assignment
    %     C: N x N self-expressive coefficient C.
    %     Y: D x N completed data.
    %     history: optimization log.

    % Set defaults.
    if nargin < 2; params = struct; end
    if nargin < 3; exprC_params = struct; end
    if nargin < 4; compY_params = struct; end
    fields = {'maxIter', 'maxTime', 'convThr', 'tauScheme', 'lambdaIncr', ...
        'lambdaIncrSteps', 'lambdaMax', 'valData', 'trueData', ...
        'prtLevel', 'logLevel'};
    defaults = {30, Inf, 1e-6, [inf inf], 1, 1, 1e4, {}, {}, 1, 1};
    for ii=1:length(fields)
      if ~isfield(params, fields{ii})
        params.(fields{ii}) = defaults{ii};
      end
    end
    exprC_params.prtLevel = params.prtLevel-1;
    exprC_params.logLevel = params.logLevel-1;
    compY_params.prtLevel = params.prtLevel-1;
    compY_params.logLevel = params.logLevel-1;
    tstart = tic; % start timer.

    % Held out observed entries for validation.
    doval = false;
    if ~isempty(params.valData)
      doval = true;
      [val_inds, Xval] = params.valData{:};
      normXval = norm(Xval); % Xval a vector of held out observed entries.
      if any(self.Omega(val_inds))
        error('Validation indices must be held out of Omega!');
      end
    end

    % True data for continuous evaluation.
    evaltrue = false;
    if ~isempty(params.trueData)
      evaltrue = true;
      [Xtrue, groupsTrue] = params.trueData{:};
      Xunobs = Xtrue(self.Omegac); normXunobs = norm(Xunobs);
    end

    % Whether tau changes across iteration.
    taus = [0 0].^params.tauScheme;
    tau_denom = max(params.maxIter-1, 1);
    % adapt_tau = ~(params.tauScheme(2)==0 || params.tauScheme(2)==inf);

    prtformstr = ['(main alt) k=%d, obj=%.2e, ' ...
        'convobj=%.2e, convC=%.2e, convY=%.2e, rtime=%.2f,%.2f'];
    if doval
      prtformstr = [prtformstr ', valerr=%.3f'];
    end
    if evaltrue
      prtformstr = [prtformstr ', cmperr=%.3f, clstrerr=%.3f, reconerr=%.3f'];
    end
    prtformstr = [prtformstr ' \n'];

    if ~all(size(Y0) == [self.D self.N])
      Y = self.X; Y(self.Omegac) = 0;
    else
      Y = Y0; Y(self.Omega) = self.X(self.Omega);
    end
    C = zeros(self.N);
    relthr = infnorm(self.X(self.Omega));
    Y_last = Y; C_last = C; obj_last = self.objective(Y, C, taus(2));
    history.status = 1;
    for kk=1:params.maxIter
      % Possibly update unobserved entry weights.
      taus = ((kk-1)/tau_denom).^params.tauScheme;
      % Deal with annoying case 1^Inf = NaN.
      taus(isnan(taus)) = 0;
      % Alternate updating C, Y.
      % Note previous iterates used to warm-start.
      [C, exprC_history] = self.exprC(Y, C, taus(1), exprC_params);
      [Y, compY_history] = self.compY(Y, C, taus(2), compY_params);

      % Diagnostic measures.
      convC = infnorm(C - C_last)/relthr;
      convY = infnorm(Y - Y_last)/relthr;
      [obj, L, R] = self.objective(Y, C, taus(2)); % Choice of tau_2 arbitrary.
      convobj = (obj_last - obj)/obj;
      val_err = []; true_scores = [];
      if doval
        val_err = norm(Y(val_inds)-Xval)/normXval;
      end
      if evaltrue
        comp_err = norm(Y(self.Omegac)-Xunobs)/normXunobs;
        [groups, ~, cluster_err] = self.cluster(C, groupsTrue);
        recon_err = sum(sum((Xtrue - Xtrue*C).^2));
        true_scores = [comp_err cluster_err recon_err];
      end

      % Printing, logging.
      if params.prtLevel > 0
        subprob_rts = [exprC_history.rtime compY_history.rtime];
        fprintf(prtformstr, [kk obj convobj convC convY subprob_rts val_err true_scores]);
      end
      if params.logLevel > 0
        history.obj(kk,:) = [obj L R];
        history.conv(kk,:) = [convobj convC convY];
        if doval
          history.val_err(kk) = val_err;
        end
        if evaltrue
          history.true_scores(kk,:) = true_scores;
        end
        if params.logLevel > 1
          history.exprC_history{kk} = exprC_history;
          history.compY_history{kk} = compY_history;
        end
      end

      % Check stopping cond: objective fails to decrease, or iterates don't change.
      if (max(convC, convY) < params.convThr) % || (~adapt_tau && convobj < params.convThr)
        history.status = 0;
        break
      end
      if toc(tstart) >= params.maxTime
        fprintf('Timeout!\n');
        break
      end
      if max(convC, convY) > 1e5
        fprintf('Divergence!\n');
        history.status = 2;
        break
      end
      C_last = C; Y_last = Y; obj_last = obj;
      if mod(kk,params.lambdaIncrSteps) == 0
        self.lambda = min(params.lambdaIncr*self.lambda, params.lambdaMax);
      end
    end
    history.iter = kk;
    if ~evaltrue
      groups = self.cluster(C);
    end
    history.rtime = toc(tstart);
    end


    function [groups, A, cluster_err] = cluster(self, C, groupsTrue)
    % cluster   Construct affinity and apply spectral clustering.
    %
    %   [groups, A, cluster_err] = solver.cluster(C, groupsTrue)
    A = build_affinity(C); % Build sparse, un-normalized affinity
    groups = spectral_clustering(A, self.n);
    if nargin > 2
      [cluster_err, groups] = eval_cluster_error(groups, groupsTrue);
    else
      cluster_err = nan;
    end
    end


  end
end
