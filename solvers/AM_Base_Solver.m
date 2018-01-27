classdef AM_Base_Solver 
% AM_Base_Solver    Base class for alternating minimization algorithms for
%   joint subspace clustering and completion.
%
%   solver = AM_Base_Solver()

  properties

  end

  methods

  function [groups, C, Y, history] = solve(self, X, Omega, n, params)
  % solve   solve alternating minimization for joint subspace clustering and
  %   completion. Minimizes a generic objective
  %
  %   min_{Y,C} 1/2 ||W_k \odot (Y - YC)||_F^2 + \lambda g(Y,C)
  %   s.t. diag(C) = 0
  %
  %   Args:
  %     X: D x N incomplete data matrix.
  %     Omega: D x N binary pattern of missing entries.
  %     n: number of clusters.
  %     params: struct containing parameters for optimization:
  %       maxIter: [default: 30].
  %       convThr: [default: 1e-6].
  %       tauScheme: scheme for updating unobserved entries of weight matrix
  %         W_k on each iteration. options are: 'fixed0', 'fixed1'
  %         (unobserved fixed to 0,1), or any number p >= 0, in which case
  %         (W_k)_{\Omega^C} = tau_k with tau_k = (k/maxIter)^p 
  %         [default: 'fixed0'].
  %       trueData: cell containing {Xtrue, groupsTrue} if available 
  %         [default: {}].
  %       prtLevel: printing level (0,1,2) [default: 1].
  %       logLevel: logging level (0,1,2) [default: 1].
  %
  %   Returns:
  %     groups: N x 1 cluster assignment
  %     C: N x N self-expressive coefficient C.
  %     Y: D x N completed data.
  %     history: optimization log.

  % Set defaults.
  fields = {'maxIter', 'convThr', 'tauScheme', 'trueData', ...
      'prtLevel', 'logLevel'};
  defaults = {30, 1e-6, 'fixed0', {}, 1, 1};
  for i=1:length(fields)
    if ~isfield(params, fields{i})
      params.(fields{i}) = defaults{i};
    end
  end
  
  % Initialize constants.
  [D, N] = size(X);
  Omega = logical(Omega); Omegac = ~Omega;
  evaltrue = false;
  if ~isempty(params.trueData)
    evaltrue = true;
    [Xtrue, groupsTrue] = params.trueData{:};
    normXtrue = norm(Xtrue, 'fro');
  end

  % Set up scheme for updating weights on unobserved entries.
  W = zeros(D, N); W(Omega) = 1; adapt_tau = false;
  if strcmpi(params.tauScheme, 'fixed1')
    W(Omegac) = 1;
  elseif isnumeric(params.tauScheme)
    adapt_tau = true;
    next_tau = @(k) (((k-1)/params.maxIter)^params.tauScheme);
  end
  
  prtformstr = ['k=%d, obj=%.2e, L=%.2e, R=%.2e, ' ...
      'convobj=%.2e, convC=%.2e, convY=%.2e'];
  if evaltrue
    prtformstr = [prtformstr ', clsterr=%.3f, cmperr=%.3f'];
  end
  prtformstr = [prtformstr ' \n'];

  Y = X; Y(Omegac) = 0; relthr = infnorm(Y);
  C = zeros(N);
  Y_last = Y; C_last = C; obj_last = inf; 
  history.status = 1;
  for kk=1:params.maxIter
    if adapt_tau
      W(Omegac) = next_tau(kk);
    end
    % Alternate updating C, Y.
    % Note previous iterates used to warm-start.
    [C, exprC_history] = self.exprC(Y, Omega, C, W, params.exprC_params);
    [Y, compY_history] = self.compY(Y, Omega, C, W, params.compY_params);
    
    % Diagnostic measures.
    convC = infnorm(C - C_last)/relthr;
    convY = infnorm(Y - Y_last)/relthr;
    [obj, L, R] = self.objective(Y, C, params);
    convobj = (obj_last - obj)/obj;
    true_scores = [];
    if evaltrue
      comp_err = norm(Y - Xtrue, 'fro')/normXtrue;
      A = build_affinity(C, true, true);
      groups = SpectralClustering(A, n);
      cluster_err = eval_cluster_error(groups, groupsTrue);
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
        history.exprC_history{k} = exprC_history;
        history.compY_history{k} = compY_history;
      end
    end
    
    % Check stopping cond: objective fails to decrease, or iterates don't change.
    if (convobj < params.convThr) || (max(convC, convY) < params.convThr);
      history.status = 0;
      break
    end

    C_last = C; Y_last = Y; obj_last = obj;
  end
  history.iter = kk;
  if ~evaltrue
    A = build_affinity(C, true, true);
    groups = SpectralClustering(A, n);
  end
  end

  end
end
