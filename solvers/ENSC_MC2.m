classdef ENSC_MC2 < ENSC_MC

  properties
  end

  methods

    function self = ENSC_MC2(X, Omega, n, lambda, gamma)
    %   Args:
    %     X: D x N incomplete data matrix.
    %     Omega: D x N binary pattern of missing entries.
    %     n: number of clusters.
    self = self@ENSC_MC(X, Omega, n, lambda, gamma);
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
    %       logLevel: 1=basic summary info, 2=detailed per-iteration info
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
    Res = Y*C - Y;
    f = self.lambda*0.5*sum(sum((W.*Res).^2));
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
