classdef ENSC_CASS_MC < ENSC_MC & CASS_MC
% ENSC_CASS_MC   Solver for combined ENSC & CASS regularized alternating
% minimization for joint subspace clustering and completion. Alternates between
%
%   min_{C} lambda/2 ||W .* (Y - YC)||_F^2 ...
%       + (gamma ||C||_1 + (1-gamma)/2||C|_F^2)
%   s.t. diag(C) = 0
%
%   min_{Y} lambda/2 ||W .* (Y - YC)||_F^2 ...
%       + \sum_{i=1}^N ||Y diag(c_i - e_i)||_*
%   s.t. P_{Omega}(Y - X) = 0
%
%   solver = ENSC_CASS_MC(X, Omega, n, lambda, gamma)

  properties
  end

  methods

    function self = ENSC_CASS_MC(X, Omega, n, lambda, gamma)
    % ENSC_CASS_MC   Solver for combined ENSC & CASS regularized alternating
    % minimization for joint subspace clustering and completion. Alternates between
    %
    %   min_{C} lambda/2 ||W .* (Y - YC)||_F^2 ...
    %       + (gamma ||C||_1 + (1-gamma)/2||C|_F^2)
    %   s.t. diag(C) = 0
    %
    %   min_{Y} lambda/2 ||W .* (Y - YC)||_F^2 ...
    %       + \sum_{i=1}^N ||Y diag(c_i - e_i)||_*
    %   s.t. P_{Omega}(Y - X) = 0
    %
    %   solver = ENSC_CASS_MC(X, Omega, n, lambda, gamma)
    %
    %   Args:
    %     X: D x N incomplete data matrix.
    %     Omega: D x N binary pattern of missing entries.
    %     n: number of clusters.
    %     lambda: self-expression penalty parameter.
    %     gamma: elastic-net tradeoff parameter.
    %
    %   Returns:
    %     self: ENSC_CASS_MC solver instance.
    self = self@ENSC_MC(X, Omega, n, lambda, gamma, 0);
    self = self@CASS_MC(X, Omega, n, lambda);
    end

    
    function [obj, L, R] = objective(self, Y, C, tau)
    % objective   Evaluate cass-elastic net objective, self-expression loss,
    %   and regularization.
    %
    %   [obj, L, R] = solver.objective(Y, C, tau)
    %
    %   Args:
    %     Y: D x N incomplete data matrix.
    %     C: N x N self-expressive coefficient C.
    %     tau: Non-negative scalar representing reconstruction penalty weight on
    %       unobserved entries.
    %
    %   Returns:
    %     obj: Objective value.
    %     L: Self-expression loss.
    %     R: CASS-Elastic net regularizer.
    Res = Y - Y*C;
    L = 0.5*sum(Res(self.Omega).^2) + 0.5*(tau^2)*sum(Res(self.Omegac).^2);
    R = self.gamma*sum(abs(C(:))) + (1-self.gamma)*0.5*sum(C(:).^2);
    CI = C - eye(self.N);
    for ii=1:self.N
      R = R + sum(svd(Y.*repmat(CI(:,ii)', [self.D 1])));
    end
    obj = self.lambda*L + R;
    end


    function [C, history] = exprC(self, Y, C, tau, params)
    % exprC   Compute self-expression with elastic-net regularization using
    %   ADMM. Solves the formulation
    %
    %   min_C \lambda/2 ||W \odot (Y - YC)||_F^2 + ...
    %     \gamma ||C||_1 + (1-gamma)/2 ||C||_F^2
    %     s.t. diag(C) = 0.
    %
    %   [C, history] = solver.exprC(Y, C, tau, params)
    %
    %   Args:
    %     Y: D x N incomplete data matrix.
    %     C: N x N self-expressive coefficient initial guess.
    %     tau: Non-negative scalar representing reconstruction penalty weight on
    %       unobserved entries.
    %     params: Struct containing problem parameters.
    %       mu: augmented Lagrangian parameter, larger values indicate smaller
    %         steps [default: 10].
    %       maxIter: [default: 200].
    %       convThr: [default: 1e-3].
    %       prtLevel: 1=basic per-iteration output [default: 0].
    %       logLevel: 0=basic summary info, 1=detailed per-iteration info
    %         [default: 0]
    %
    %   Returns:
    %     C: N x N self-expression.
    %     history: Struct containing diagnostic info.
    [C, history] = exprC@ENSC_MC(self, Y, C, tau, params);
    end

    
    function [Y, history] = compY(self, Y, C, tau, params)
    % compY   Complete missing data using self-expression C. Solves the
    %   objective
    %
    %   min_Y \lambda/2 ||W .* (Y - YC)||_F^2 + ...
    %     \sum_{i=1}^N ||Y diag(c_i - e_i)||_*
    %
    %   using admm by introducing auxiliary variables L_i = Y diag(c_i - e_i)
    %
    %   [Y, history] = solver.compY(Y, C, tau, params)
    %
    %   Args:
    %     Y: D x N complete data matrix initial guess.
    %     C: N x N self-expressive coefficient C.
    %     tau: Non-negative scalar representing reconstruction penalty weight on
    %       unobserved entries.
    %     params: Struct containing problem parameters.
    %       mu: augmented Lagrangian parameter, larger values indicate smaller
    %         steps [default: 10].
    %       maxIter: [default: 200].
    %       convThr: [default: 1e-3].
    %       prtLevel: 1=basic per-iteration output [default: 0].
    %       logLevel: 0=basic summary info, 1=detailed per-iteration info
    %         [default: 0]
    %
    %   Returns:
    %     Y: D x N completed data.
    %     history: Struct containing minimal diagnostic info.
    [Y, history] = compY@CASS_MC(self, Y, C, tau, params);
    % [Y, history] = compY@CASS_MC(self, Y, abs(C)'*abs(C), tau, params);
    end
    
    
    function lambda = adapt_lambda(self, alpha, Y, tau)
    % adapt_lambda    Compute lambda as alpha*lambda_min where c_i = 0 is a
    %   solution for some i iff lambda <= lambda_min.
    %
    %   solver = solver.adapt_lambda(alpha, Y, tau)
    %
    %   Args:
    %     alpha: penalty parameter > 1.
    %     Y: D x N data matrix.
    %     tau: Non-negative scalar representing reconstruction penalty weight on
    %       unobserved entries.
    %
    %   Returns:
    %     lambda: adapted lambda.
    lambda = adapt_lambda@ENSC_MC(self, alpha, Y, tau);
    end

  end

end
