classdef ENSC_MC_apg < ENSC_MC
% ENSC_MC_apg   Solver for elastic-net regularized alternating minimization for
%   joint subspace clustering and completion. Solves formulation
%
%   min_{Y,C} lambda/2 ||W .* (Y - YC)||_F^2 ...
%       + gamma ||C||_1 + (1-gamma)/2 ||C||_F^2 ...
%       + eta/2 ||P_{Omega^c}(Y)||_F^2
%   s.t. diag(C) = 0, P_{Omega}(Y - X) = 0
%
%   Uses accelerated prox grad (fista), rather than admm to solve
%   self-expression problem.
%
%   solver = ENSC_MC_apg(X, Omega, n, lambda, gamma, eta)

  properties
  end

  methods

    function self = ENSC_MC_apg(X, Omega, n, lambda, gamma, eta)
    % ENSC_MC_apg   Solver for elastic-net regularized alternating minimization for
    %   joint subspace clustering and completion. Solves formulation
    %
    %   min_{Y,C} lambda/2 ||W .* (Y - YC)||_F^2 ...
    %       + gamma ||C||_1 + (1-gamma)/2 ||C||_F^2 ...
    %       + eta/2 ||P_{Omega^c}(Y)||_F^2
    %   s.t. diag(C) = 0, P_{Omega}(Y - X) = 0
    %
    %   Uses accelerated prox grad (fista), rather than admm to solve
    %   self-expression problem.
    %
    %   solver = ENSC_MC_apg(X, Omega, n, lambda, gamma, eta)
    %
    %   Args:
    %     X: D x N incomplete data matrix.
    %     Omega: D x N binary pattern of missing entries.
    %     n: number of clusters.
    %     lambda: self-expression penalty parameter.
    %     gamma: elastic-net tradeoff parameter.
    %     eta: frobenius penalty parameter on completion.
    %
    %   Returns:
    %     self: ENSC_MC solver instance.
    if nargin < 6; eta = 0; end
    self = self@ENSC_MC(X, Omega, n, lambda, gamma, eta);
    end


    function [C, history] = exprC(self, Y, C0, tau, params)
    % exprC   Compute self-expression with elastic-net regularization using
    %   accelerated proximal gradient.  Solves the formulation
    %
    %   min_C \lambda/2 ||W \odot (Y - YC)||_F^2 + ...
    %     \gamma ||C||_1 + (1-gamma)/2 ||C||_F^2
    %     s.t. diag(C) = 0.
    %
    %   [C, history = solver.exprC(Y, C0, tau, params)
    %
    %   Args:
    %     Y: D x N incomplete data matrix.
    %     C0: N x N self-expressive coefficient initial guess.
    %     tau: Non-negative scalar representing reconstruction penalty weight on
    %       unobserved entries.
    %     params: Struct containing problem parameters.
    %       maxIter: [default: 500].
    %       convThr: [default: 1e-4].
    %       prtLevel: 1=basic per-iteration output [default: 0].
    %       logLevel: 0=basic summary info, 1=detailed per-iteration info
    %         [default: 0]
    %
    %   Returns:
    %     C: N x N self-expression.
    %     history: Struct containing diagnostic info.

    % Set defaults.
    fields = {'maxIter', 'convThr', 'prtLevel', 'logLevel'};
    defaults = {500, 1e-4, 0, 0};
    for i=1:length(fields)
      if ~isfield(params, fields{i})
        params.(fields{i}) = defaults{i};
      end
    end

    W = ones(self.D, self.N); W(self.Omegac) = tau;
    function [f, G] = exprC_ffun(C)
    Res = W.*(Y*C - Y);
    f = self.lambda*0.5*sum(sum(Res.^2));
    if nargout > 1
      G = self.lambda*(Y'*(W.*Res));
    end
    end

    function [r, Z] = exprC_rfun(C, rho)
    r = self.gamma*sum(abs(C(:))) + 0.5*(1-self.gamma)*sum(C(:).^2);
    if nargout > 1
      Z = prox_en(C, self.gamma, rho);
      Z(1:(self.N+1):end) = 0; % set diagonal to zero.
    end
    end

    [C, history] = apg(C0, @exprC_ffun, @exprC_rfun, params);
    end

    end

end
