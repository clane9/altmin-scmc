classdef LSRSC_MC < ENSC_MC
% LSRSC_MC   Solver for least-squares regularized alternating minimization for
%   joint subspace clustering and completion. Solves formulation
%
%   min_{Y,C} lambda/2 ||W .* (Y - YC)||_F^2 ...
%       + 1/2 ||C||_F^2 ...
%       + eta/2 ||P_{Omega^c}(Y)||_F^2
%   s.t. diag(C) = 0, P_{Omega}(Y - X) = 0
%
%   solver = LSRSC_MC(X, Omega, n, lambda, eta)

  methods

    function self = LSRSC_MC(X, Omega, n, lambda, eta)
    % LSRSC_MC   Solver for least-squares regularized alternating minimization for
    %   joint subspace clustering and completion. Solves formulation
    %
    %   min_{Y,C} lambda/2 ||W .* (Y - YC)||_F^2 ...
    %       + 1/2 ||C||_F^2 ...
    %       + eta/2 ||P_{Omega^c}(Y)||_F^2
    %   s.t. diag(C) = 0, P_{Omega}(Y - X) = 0
    %
    %   solver = LSRSC_MC(X, Omega, n, lambda, eta)
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
    %     self: LSRSC_MC solver instance.
    if nargin < 5; eta = 0; end
    self = self@ENSC_MC(X, Omega, n, 0, lambda, eta);
    end


    function [obj, L, R] = objective(self, Y, C, tau)
    % objective   Evaluate elastic net objective, self-expression loss, and
    %   regularization.
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
    %     R: Elastic net regularizer.
    Res = Y - Y*C;
    L = 0.5*sum(Res(self.Omega).^2) + 0.5*(tau^2)*sum(Res(self.Omegac).^2);
    R = 0.5*sum(C(:).^2);
    if self.eta > 0
      R = R + self.eta*0.5*sum(Y(self.Omegac).^2);
    end
    obj = self.lambda*L + R;
    end


    function [C, history] = exprC(self, Y, ~, tau, ~)
    % exprC   Compute self-expression with least-squares regularization.
    %   Solves the formulation
    %
    %   min_C \lambda/2 ||W \odot (Y - YC)||_F^2 + 1/2 ||C||_F^2
    %     s.t. diag(C) = 0.
    %
    %   [C, history] = solver.exprC(Y, C, tau, params)
    %
    %   Args:
    %     Y: D x N incomplete data matrix.
    %     C: N x N self-expressive coefficient initial guess.
    %     tau: Non-negative scalar representing reconstruction penalty weight on
    %       unobserved entries.
    %     params: Not used, included for consistency.
    %
    %   Returns:
    %     C: N x N self-expression.
    %     history: Struct containing diagnostic info.
    tstart = tic; % start timer.
    W = ones(self.D, self.N); W(self.Omegac) = tau;
    C = zeros(self.N); negI = ~logical(eye(self.N));
    for ii=1:self.N
      % Compute least squares solution to:
      %   lambda/2 ||diag(W_i) (Y c_i - y_i)||_2^2 + 1/2 ||c_i||_2^2
      %   s.t. c_ii = 0
      wi = W(:,ii); negIi = negI(:,ii);
      wY = ldiagmult(wi, Y); wyi = wY(:,ii); wY = trimmat(wY, ii);
      C(:, negIi) = (self.lambda*(wY'*wY) + eye(self.N-1)) \ ...
          (self.lambda*(wY'*wyi));
    end
    history.iter = 0; history.status = 0; history.rtime = toc(tstart);
    end


    function lambda = adapt_lambda(~, alpha, ~, ~)
    % adapt_lambda    Only lambda = 0 will result in zero solution for this
    %   formulation, making this function unnecessary. Only included for
    %   consistency.
    %
    %   solver = solver.adapt_lambda(alpha, Y, tau)
    %
    %   Args:
    %     alpha: penalty parameter > 0.
    %     Y: D x N data matrix (not used).
    %     tau: Non-negative scalar representing reconstruction penalty weight on
    %       unobserved entries (not used).
    %
    %   Returns:
    %     lambda: adapted lambda (lambda = alpha).
    lambda = alpha;
    end

  end

end
