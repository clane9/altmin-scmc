classdef ENSC_MC_spams_comp_apg < ENSC_MC_spams
% ENSC_MC_spams_comp_apg   Solver for elastic-net regularized alternating minimization for
%   joint subspace clustering and completion. Uses APG instead of closed form
%   solution for completion. Solves formulation
%
%   min_{Y,C} lambda/2 ||W .* (Y - YC)||_F^2 ...
%       + gamma ||C||_1 + (1-gamma)/2 ||C||_F^2 ...
%       + eta/2 ||P_{Omega^c}(Y)||_F^2
%   s.t. diag(C) = 0, P_{Omega}(Y - X) = 0
%
%   using SPAMS off-the-shelf solver, rather than admm to solve
%   self-expression problem. Uses APG rather than closed form solution for completion.
%
%   solver = ENSC_MC_spams_comp_apg(X, Omega, n, lambda, gamma, eta)

  properties
  end

  methods

    function self = ENSC_MC_spams_comp_apg(X, Omega, n, lambda, gamma, eta)
    % ENSC_MC_spams_comp_apg   Solver for elastic-net regularized alternating minimization for
    %   joint subspace clustering and completion. Solves formulation
    %
    %   min_{Y,C} lambda/2 ||W .* (Y - YC)||_F^2 ...
    %       + gamma ||C||_1 + (1-gamma)/2 ||C||_F^2 ...
    %       + eta/2 ||P_{Omega^c}(Y)||_F^2
    %   s.t. diag(C) = 0, P_{Omega}(Y - X) = 0
    %
    %   Uses SPAMS off-the-shelf solver, rather than admm to solve
    %   self-expression problem. Uses APG rather than closed form solution for completion.
    %
    %   solver = ENSC_MC_spams_comp_apg(X, Omega, n, lambda, gamma, eta)
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
    self = self@ENSC_MC_spams(X, Omega, n, lambda, gamma, eta);
    end


    function [Y, history] = compY(self, Y0, C, tau, params)
    % compY   Complete missing data using self-expression C. Solves objective:
    %
    %   min_Y lambda/2 ||W \odot (Y - YC)||_F^2 + eta/2 ||P_\Omega^C(Y)||_F^2
    %   s.t. P_Omega(Y-X) = 0.
    %
    %   Problem is solved using APG.
    %
    %   [Y, history] = solver.compY(Y, C, tau, params)
    %
    %   Args:
    %     Y0: D x N complete data matrix initial guess
    %     C: N x N self-expressive coefficient C.
    %     tau: Non-negative scalar representing reconstruction penalty weight on
    %       unobserved entries.
    %     params: optimization parameters for APG.
    %       maxIter: [default: 500].
    %       convThr: [default: 1e-4].
    %       prtLevel: 1=basic per-iteration output [default: 0].
    %       logLevel: 0=basic summary info, 1=detailed per-iteration info
    %         [default: 0]
    %
    %   Returns:
    %     Y: D x N completed data.
    %     history: Struct containing minimal diagnostic info.

    % Set defaults.
    fields = {'maxIter', 'convThr', 'prtLevel', 'logLevel'};
    defaults = {500, 1e-4, 0, 0};
    for i=1:length(fields)
      if ~isfield(params, fields{i})
        params.(fields{i}) = defaults{i};
      end
    end

    W = ones(self.D, self.N); W(self.Omegac) = tau;
    IC = eye(self.N) - C; ICt = IC';
    Xobs = self.X(self.Omega);
    
    function [f, G] = compY_ffun(Y)
    Res = W.*(Y*IC);
    f = self.lambda*0.5*sum(sum(Res.^2));
    if self.eta > 0
      f = f + self.eta*0.5*sum(sum(Y.^2));
    end
    if nargout > 1
      G = self.lambda*((W.*Res)*ICt);
      if self.eta > 0
        G = G + self.eta*Y;
      end
    end
    end

    function [r, Z] = compY_rfun(Y, ~)
    r = 0;
    if nargout > 1
      Z = Y; Z(self.Omega) = Xobs;
    end
    end

    [Y, history] = apg(Y0, @compY_ffun, @compY_rfun, params);
    end


    end

end
