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

    function [groups, C, Y, history] = solve(self, params, exprC_params, ...
        compY_params)
    % solve   solve alternating minimization for joint subspace clustering and
    %   completion. Minimizes a generic objective
    %
    %   min_{Y,C} lambda/2 ||W_k \odot (Y - YC)||_F^2 + g(Y,C)
    %   s.t. diag(C) = 0
    %
    %   [groups, C, Y, history] = solver.solve(params, exprC_params, compY_params)
    %
    %   Args:
    %     params: struct containing parameters for optimization:
    %       maxIter: [default: 30].
    %       convThr: [default: 1e-6].
    %       tauScheme: scheme for updating unobserved entries of weight matrix
    %         W_k on each iteration. options are: 'fixed0', 'fixed1'
    %         (unobserved fixed to 0,1), 'switch01' (0 for self-express, 1 for
    %         completion), or any number p >= 0, in which case (W_k)_{\Omega^C}
    %         = tau_k with tau_k = (k/maxIter)^p [default: 'fixed0'].
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
    fields = {'maxIter', 'convThr', 'tauScheme', 'trueData', ...
        'prtLevel', 'logLevel'};
    defaults = {30, 1e-6, 'fixed0', {}, 1, 1};
    for ii=1:length(fields)
      if ~isfield(params, fields{ii})
        params.(fields{ii}) = defaults{ii};
      end
    end
    exprC_params.prtLevel = params.prtLevel-1;
    exprC_params.logLevel = params.logLevel-1;
    compY_params.prtLevel = params.prtLevel-1;
    compY_params.logLevel = params.logLevel-1;
    tic; % start timer.

    % Initialize constants.
    evaltrue = false;
    if ~isempty(params.trueData)
      evaltrue = true;
      [Xtrue, groupsTrue] = params.trueData{:};
      Xunobs = Xtrue(self.Omegac); normXunobs = norm(Xunobs);
    end

    % Set up scheme for updating weights on unobserved entries.
    tau = 0; exprC_tau = 0; adapt_tau = false;
    if strcmpi(params.tauScheme, 'fixed1')
      tau = 1; exprC_tau = 1;
    elseif strcmpi(params.tauScheme, 'switch01') || strcmpi(params.tauScheme, 'firstexpr0')
      tau = 1; exprC_tau = 0;
    elseif isnumeric(params.tauScheme)
      adapt_tau = true;
      next_tau = @(k) (((k-1)/params.maxIter)^params.tauScheme);
    end

    prtformstr = ['(main alt) k=%d, obj=%.2e, L=%.2e, R=%.2e, ' ...
        'convobj=%.2e, convC=%.2e, convY=%.2e'];
    if evaltrue
      prtformstr = [prtformstr ', cmperr=%.3f, clstrerr=%.3f'];
    end
    prtformstr = [prtformstr ' \n'];

    Y = self.X; Y(self.Omegac) = 0; relthr = infnorm(self.X(self.Omega));
    C = zeros(self.N);
    Y_last = Y; C_last = C; obj_last = self.objective(Y, C, tau);
    history.status = 1;
    for kk=1:params.maxIter
      if adapt_tau
        tau = next_tau(kk); exprC_tau = tau;
      end
      if kk==2 && strcmpi(params.tauScheme, 'firstexpr0')
        exprC_tau = 1;
      end
      % Alternate updating C, Y.
      % Note previous iterates used to warm-start.
      [C, exprC_history] = self.exprC(Y, C, exprC_tau, exprC_params);
      [Y, compY_history] = self.compY(Y, C, tau, compY_params);

      % Diagnostic measures.
      convC = infnorm(C - C_last)/relthr;
      convY = infnorm(Y - Y_last)/relthr;
      [obj, L, R] = self.objective(Y, C, tau);
      convobj = (obj_last - obj)/obj;
      true_scores = [];
      if evaltrue
        comp_err = norm(Y(self.Omegac) - Xunobs)/normXunobs;
        [groups, ~, cluster_err] = self.cluster(C, groupsTrue);
        true_scores = [comp_err cluster_err];
      end

      % Printing, logging.
      if params.prtLevel > 0
        fprintf(prtformstr, [kk obj L R convobj convC convY true_scores]);
      end
      if params.logLevel > 0
        history.obj(kk,:) = [obj L R];
        history.conv(kk,:) = [convobj convC convY];
        if evaltrue
          history.true_scores(kk,:) = true_scores;
        end
        if params.logLevel > 1
          history.exprC_history{kk} = exprC_history;
          history.compY_history{kk} = compY_history;
        end
      end

      % Check stopping cond: objective fails to decrease, or iterates don't change.
      if (~adapt_tau && convobj < params.convThr) || (max(convC, convY) < params.convThr);
        history.status = 0;
        break
      end

      C_last = C; Y_last = Y; obj_last = obj;
    end
    history.iter = kk;
    if ~evaltrue
      groups = self.cluster(C);
    end
    history.rtime = toc;
    end


    function [groups, A, cluster_err] = cluster(self, C, groupsTrue)
    % cluster   Construct affinity and apply spectral clustering.
    %
    %   [groups, A, cluster_err] = solver.cluster(C, groupsTrue)
    A = build_affinity(C); % Build sparse, un-normalized affinity
    groups = SpectralClustering(A, self.n);
    if nargin > 2
      [cluster_err, groups] = eval_cluster_error(groups, groupsTrue);
    else
      cluster_err = nan;
    end
    end

  end
end
