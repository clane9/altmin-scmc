classdef ENSC_MC < SC_MC_Base_Solver

  properties
    gamma;
  end

  methods

    function self = ENSC_MC(X, Omega, n, lambda, gamma)
    %   Args:
    %     X: D x N incomplete data matrix.
    %     Omega: D x N binary pattern of missing entries.
    %     n: number of clusters.
    self = self@SC_MC_Base_Solver(X, Omega, n, lambda);
    self.gamma = gamma;
    end


    function [obj, L, R] = objective(self, Y, C, tau)
    % objective   Evaluate elastic net objective, self-expression loss, and
    %   regularization.
    %
    %   [obj, L, R] = solver.objective(Y, C, tau)
    %
    %   Args:
    %     Y: D x N incomplete data matrix.
    %     C: N x N self-expressive coefficient C, used for initialization.
    %     tau: Non-negative scalar representing reconstruction penalty weight on
    %       unobserved entries.
    %
    %   Returns:
    %     obj: Objective value.
    %     L: Self-expression loss.
    %     R: Elastic net regularizer.
    Res = Y - Y*C;
    L = 0.5*sum(Res(self.Omega).^2) + 0.5*(tau^2)*sum(Res(self.Omegac).^2);
    R = self.gamma*sum(abs(C(:))) + (1-self.gamma)*0.5*sum(C(:).^2);
    obj = self.lambda*L + R;
    end


    function [C, history] = exprC(self, Y, C, tau, params)
    % exprC   Compute self-expression with elastic-net regularization using ADMM.
    %   Solves the formulation
    %
    %   min_C \lambda/2 ||W \odot (Y - YC)||_F^2 + ...
    %     \gamma ||C||_1 + (1-gamma)/2 ||C||_F^2
    %     s.t. diag(C) = 0.
    %
    %   [C, history = solver.exprC(Y, C, tau, params)
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
    %       convThr: [default: 1e-4].
    %       prtLevel: 1=basic per-iteration output [default: 0].
    %       logLevel: 1=basic summary info, 2=detailed per-iteration info
    %         [default: 1]
    %
    %   Returns:
    %     C: N x N self-expression.
    %     history: Struct containing diagnostic info.

    % Set defaults.
    fields = {'mu', 'maxIter', 'convThr', 'prtLevel', 'logLevel'};
    defaults = {10, 200, 1e-4, 0, 0};
    for i=1:length(fields)
      if ~isfield(params, fields{i})
        params.(fields{i}) = defaults{i};
      end
    end
    tic; % start timer.

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
      % Update C by solving elastic-net proximal operator, with diagonal constraint.
      C = prox_en(Z + U, self.gamma, 1/params.mu);
      C(1:(self.N+1):end) = 0; % set diagonal to 0.
      % Update variables used to absorb errors on \Omega, \Omega^c.
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
    history.iter = kk;
    history.rtime = toc;
    end


    function [Y, history] = compY(self, ~, C, tau, ~)
    % compY   Complete missing data using self-expression C. Solves objective:
    %
    %   min_Y 1/2 ||W \odot (Y - YC)||_F^2 s.t. P_Omega(Y-X) = 0.
    %
    %   Problem is solved row-by-row by computing a least-squares solution
    %   using SVD.
    %
    %   [Y, history] = solver.compY(~, C, tau, ~)
    %
    %   Args:
    %     Y: D x N complete data matrix initial guess (not used, included for
    %       consistency).
    %     C: N x N self-expressive coefficient C.
    %     tau: Non-negative scalar representing reconstruction penalty weight on
    %       unobserved entries.
    %
    %   Returns:
    %     Y: D x N completed data.
    %     history: Struct containing minimal diagnostic info.
    tic; % start timer.
    W = ones(self.D, self.N); W(self.Omegac) = tau;
    IC = eye(self.N) - C;
    Y = self.X; % Initialize Y so that it agrees on observed entries.
    for ii=1:self.D
      omegai = self.Omega(ii,:); omegaic = self.Omegac(ii,:);
      xi = self.X(ii,:)'; wi = W(ii,:)';
      % Compute A = ((I - C) diag(W_{i,.}))^T. Drop rows of A set to zero.
      A = IC' .* repmat(wi, [1 self.N]); A = A(wi~=0, :);
      % Compute least squares solution to:
      %   1/2 ||A_{\omega_i^c} y_{\omega_i^c} + A_{\omega_i} x_{\omega_i}||_2^2
      Y(ii,omegaic) = pinv(A(:,omegaic))*(-A(:,omegai)*xi(omegai));
    end
    history.iter = 0; history.status = 0; history.rtime = toc;
    end

    end

end
