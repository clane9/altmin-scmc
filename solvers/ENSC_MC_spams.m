classdef ENSC_MC_spams < ENSC_MC
% ENSC_MC_spams   Solver for elastic-net regularized alternating minimization for
%   joint subspace clustering and completion. Solves formulation
%
%   min_{Y,C} lambda/2 ||W .* (Y - YC)||_F^2 ...
%       + gamma ||C||_1 + (1-gamma)/2 ||C||_F^2 ...
%       + eta/2 ||P_{Omega^c}(Y)||_F^2
%   s.t. diag(C) = 0, P_{Omega}(Y - X) = 0
%
%   using SPAMS off-the-shelf solver, rather than admm to solve
%   self-expression problem.
%
%   solver = ENSC_MC_spams(X, Omega, n, lambda, gamma, eta)

  properties
  end

  methods

    function self = ENSC_MC_spams(X, Omega, n, lambda, gamma, eta)
    % ENSC_MC_spams   Solver for elastic-net regularized alternating minimization for
    %   joint subspace clustering and completion. Solves formulation
    %
    %   min_{Y,C} lambda/2 ||W .* (Y - YC)||_F^2 ...
    %       + gamma ||C||_1 + (1-gamma)/2 ||C||_F^2 ...
    %       + eta/2 ||P_{Omega^c}(Y)||_F^2
    %   s.t. diag(C) = 0, P_{Omega}(Y - X) = 0
    %
    %   Uses SPAMS off-the-shelf solver, rather than admm to solve
    %   self-expression problem.
    %
    %   solver = ENSC_MC_spams(X, Omega, n, lambda, gamma, eta)
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


    function [C, history] = exprC(self, Y, ~, tau, ~)
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
    %     C: N x N self-expressive coefficient initial guess (not used).
    %     tau: Non-negative scalar representing reconstruction penalty weight on
    %       unobserved entries.
    %     params: (not used, included for consistency.)
    %
    %   Returns:
    %     C: N x N self-expression.
    %     history: Struct containing diagnostic info.
    tstart = tic; % start timer.
    W = ones(self.D, self.N); W(self.Omegac) = tau;
    C = zeros(self.N);
    % Convert to spams notation.
    spams_param.lambda = self.gamma/self.lambda;
    spams_param.lambda2 = (1-self.gamma)/self.lambda;
    spams_param.numThreads = 4;
    for ii=1:self.N
      wi = W(:,ii);
      x = wi.*Y(:,ii);
      D = ldiagmult(wi, trimmat(Y, ii));
      c = mexLasso(x, D, spams_param);
      C(:,ii) = [c(1:(ii-1)); 0; c(ii:end)];
    end
    history.iter = 0; history.status = 0; history.rtime = toc(tstart);
    end

    end

end
