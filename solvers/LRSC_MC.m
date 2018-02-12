classdef LRSC_MC < ENSC_MC
% LRSC_MC   Solver for low-rank regularized alternating minimization for
%   joint subspace clustering and completion. Solves formulation
%
%   min_{Y,C} lambda/2 ||W .* (Y - YC)||_F^2 ...
%       + ||C||_* + eta/2 ||P_{Omega^c}(Y)||_F^2
%   s.t. P_{Omega}(Y - X) = 0
%
%   solver = LRSC_MC(X, Omega, n, lambda, eta)

  methods

    function self = LRSC_MC(X, Omega, n, lambda, eta)
    % LRSC_MC   Solver for low-rank regularized alternating minimization for
    %   joint subspace clustering and completion. Solves formulation
    %
    %   min_{Y,C} lambda/2 ||W .* (Y - YC)||_F^2 ...
    %       + ||C||_* + eta/2 ||P_{Omega^c}(Y)||_F^2
    %   s.t. P_{Omega}(Y - X) = 0
    %
    %   solver = LRSC_MC(X, Omega, n, lambda, eta)
    %
    %   Args:
    %     X: D x N incomplete data matrix.
    %     Omega: D x N binary pattern of missing entries.
    %     n: number of clusters.
    %     lambda: self-expression penalty parameter.
    %     eta: frobenius penalty parameter on completion.
    %
    %   Returns:
    %     self: LRSC_MC solver instance.
    if nargin < 5; eta = 0; end
    self = self@ENSC_MC(X, Omega, n, lambda, 0, eta);
    end


    function [obj, L, R] = objective(self, Y, C, tau)
    % objective   Evaluate LRSC objective, self-expression loss, and
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
    R = sum(svd(C));
    if self.eta > 0
      R = R + self.eta*0.5*sum(Y(self.Omegac).^2);
    end
    obj = self.lambda*L + R;
    end


    function [C, history] = exprC(self, Y, C, tau, params)
    % exprC   Compute self-expression with low-rank regularization using
    %   ADMM. Solves the formulation
    %
    %   min_C \lambda/2 ||W \odot (Y - YC)||_F^2 + ||C||_*
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

    % Set defaults.
    fields = {'mu', 'maxIter', 'convThr', 'prtLevel', 'logLevel'};
    defaults = {10, 200, 1e-3, 0, 0};
    for i=1:length(fields)
      if ~isfield(params, fields{i})
        params.(fields{i}) = defaults{i};
      end
    end
    tstart = tic; % start timer.

    % Set constants.
    tausqr = tau^2;
    YtY = Y'*Y;
    R = chol(self.lambda*(tausqr+1)*YtY + params.mu*eye(self.N));
    relthr = infnorm(Y(self.Omega));

    % Initialize variables.
    Res = Y*C - Y; A = self.Omegac .* Res; B = self.Omega .* Res;
    U = zeros(self.N); % scaled Lagrange multiplier

    prtformstr = 'k=%d, obj=%.2e, feas=%.2e \n';

    history.status = 1;
    for kk=1:params.maxIter
      % Update Z (proxy for C) by least squares.
      leastsqr_target = self.lambda*YtY + params.mu*(C - U);
      if tau < 1
        leastsqr_target = leastsqr_target + self.lambda*Y'*A;
        if tau > 0
          leastsqr_target = leastsqr_target + self.lambda*tausqr*(YtY + Y'*B);
        end
      end
      Z = R \ (R' \ leastsqr_target);
      % Update C by nuclear norm proximal operator.
      C = prox_nuc(Z + U, 1/params.mu);
      % Update A, B variables used to absorb errors on \Omega, \Omega^c.
      Res = Y*Z - Y;
      if tau < 1
        A = self.Omegac .* Res;
        if tau > 0
          B = self.Omega .* Res;
        end
      end
      % Update scaled Lagrange multiplier.
      U = U + (Z - C);

      % Diagnostic measures, printing, logging.
      feas = infnorm(Z - C)/relthr;
      obj = self.objective(Y, C, tau);
      if params.prtLevel > 0
        fprintf(prtformstr, kk, obj, feas);
      end
      if params.logLevel > 0
        history.obj(kk) = obj;
        history.feas(kk) = feas;
      end

      if feas < params.convThr
        history.status = 0;
        break
      end
    end
    history.iter = kk; history.rtime = toc(tstart);
    end

    function lambda = adapt_lambda(self, alpha, Y, tau)
    % adapt_lambda    Compute lambda as alpha*lambda_min where C = 0 is a
    %   solution for some iff lambda <= lambda_min.
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
    W = ones(self.D, self.N); W(self.Omegac) = tau;
    lambdamin = 1 / norm(Y'*(W.^2 .* Y), 2);
    lambda = alpha*lambdamin;
    end

    end

end
