classdef CASS_MC < SC_MC_Base_Solver
% CASS_MC   Solver for CASS regularized alternating minimization for
%   joint subspace clustering and completion. Solves formulation
%
%   min_{Y,C} lambda/2 ||W .* (Y - YC)||_F^2 ...
%       + \sum_{i=1}^N ||Y diag(c_i - e_i)||_*
%   s.t. diag(C) = 0, P_{Omega}(Y - X) = 0
%
%   solver = CASS_MC(X, Omega, n, lambda)

  properties
  end

  methods

    function self = CASS_MC(X, Omega, n, lambda)
    % CASS_MC   Solver for CASS regularized alternating minimization for
    %   joint subspace clustering and completion. Solves formulation
    %
    %   min_{Y,C} lambda/2 ||W .* (Y - YC)||_F^2 ...
    %       + \sum_{i=1}^N ||Y diag(c_i - e_i)||_*
    %   s.t. diag(C) = 0, P_{Omega}(Y - X) = 0
    %
    %   using admm by introducing auxiliary variables L_i = Y diag(c_i - e_i)
    %
    %   solver = CASS_MC(X, Omega, n, lambda)
    %
    %   Args:
    %     X: D x N incomplete data matrix.
    %     Omega: D x N binary pattern of missing entries.
    %     n: number of clusters.
    %     lambda: self-expression penalty parameter.
    %
    %   Returns:
    %     self: CASS_MC solver instance.
    self = self@SC_MC_Base_Solver(X, Omega, n, lambda);
    end


    function [obj, L, R] = objective(self, Y, C, tau)
    % objective   Evaluate CASS objective, self-expression loss, and
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
    %     R: CASS regularizer.
    Res = Y - Y*C;
    L = 0.5*sum(Res(self.Omega).^2) + 0.5*(tau^2)*sum(Res(self.Omegac).^2);
    CI = C - eye(self.N);
    R = 0;
    for ii=1:self.N
      R = R + sum(svd(Y.*repmat(CI(:,ii)', [self.D 1])));
    end
    obj = self.lambda*L + R;
    end


    function [C, history] = exprC(self, Y, C, tau, params)
    % exprC   Compute self-expression with CASS regularization using
    %   ADMM. Solves the formulation
    %
    %   min_C \lambda/2 ||W .* (Y - YC)||_F^2 + ...
    %     \sum_{i=1}^N ||Y diag(c_i - e_i)||_*
    %     s.t. diag(C) = 0.
    %
    %   [C, history] = solver.exprC(Y, C, tau, params)
    %
    %   using admm by introducing auxiliary variables L_i = Y diag(c_i - e_i)
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
    W = ones(self.D, self.N); W(self.Omegac) = tau;
    relthr = infnorm(Y(self.Omega));
    repY = repmat(Y, [1 1 self.N]);

    % Pre-compute Cholesky factorizations for C updates.
    RR = zeros(self.N-1, self.N-1, self.N);
    for ii=1:self.N
      A = ldiagmult(W(:,ii), Y);
      Atrim = trimmat(A, ii);
      Ytrim = trimmat(Y, ii);
      M = self.lambda*(Atrim'*Atrim) + params.mu*diag(sum((Ytrim.^2)));
      RR(:,:,ii) = chol(M);
    end

    % Initialize variables.
    [LL, ULL] = deal(zeros(self.D, self.N, self.N));
    CI = C; CI(1:(self.N+1):end) = -1;
    repCI = repmat(CI, [1 1 self.D]);
    repCI = permute(repCI, [3 1 2]);
    allYdiagCI = repY.*repCI;
    Lrnks = zeros(self.N, 1);

    prtformstr = 'k=%d, feasLL=%.2e, minr=%d, maxr=%d, rt=%.2f \n';

    history.status = 1;
    for kk=1:params.maxIter
      itertstart = tic;
      % Update each of the N L^i by SVD shrinkage-thresholding (expensive!!).
      QQ = allYdiagCI - ULL;
      for ii=1:self.N
        Q = QQ(:,:,ii);
        Qnorm = sqrt(sum(Q.^2));
        nzmask = Qnorm > 1e-5*max(Qnorm); % Often entire columns will be zero.
        [LL(:,nzmask,ii), Lrnks(ii)] = prox_nuc(Q(:,nzmask), 1/params.mu);
        % [LL(:,nzmask,ii), Lrnks(ii)] = prox_nuc(Q(:,nzmask), 1/params.mu, Lrnks(ii));
      end
      % Update each c_i separately by solving a least-squares problem.
      for ii=1:self.N
        A = ldiagmult(W(:,ii), Y);
        a = A(:, ii); Atrim = trimmat(A, ii);
        Ytrim = trimmat(Y, ii);
        b = self.lambda*(Atrim'*a) + ...
            params.mu*sum(Ytrim.*trimmat(LL(:,:,ii) + ULL(:,:,ii), ii))';
        c = RR(:,:,ii) \ (RR(:,:,ii)' \ b);
        CI(:, ii) = [c(1:(ii-1)); -1; c(ii:end)];
      end
      % ith slice of repC contains 1 c_i^T.
      % Used for multiplying slices of DxNxN tensor by diag(c_i).
      repCI = repmat(CI, [1 1 self.D]);
      repCI = permute(repCI, [3 1 2]);
      allYdiagCI = repY.*repCI;
      % Update scaled Lagrange multiplier.
      conLL = LL - allYdiagCI;
      ULL = ULL + conLL;

      % Diagnostic measures, printing, logging.
      feas = infnorm(conLL)/relthr;
      if params.prtLevel > 0
        fprintf(prtformstr, kk, feas, min(Lrnks), max(Lrnks), toc(itertstart));
      end
      if params.logLevel > 0
        history.feas(kk) = feas;
        history.Lrnk(kk,:) = [min(Lrnks) median(Lrnks) max(Lrnks)];
      end

      if feas < params.convThr
        history.status = 0;
        break
      end
    end
    C = CI; C(1:(self.N+1):end) = 0;
    history.iter = kk; history.rtime = toc(tstart);
    end


    function [Y, history] = compY(self, Y, C, tau, params)
    % compY   Complete missing data using self-expression C. Solves objective:
    %
    %   min_Y \lambda/2 ||W .* (Y - YC)||_F^2 + ...
    %     \sum_{i=1}^N ||Y diag(c_i - e_i)||_*
    %
    %   using admm by introducing auxiliary variables L_i = Y diag(c_i - e_i)
    %
    %   [Y, history] = solver.compY(Y, C, tau, params)
    %
    %   Args:
    %     Y: D x N complete data matrix initial guess (not used, included for
    %       consistency).
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
    W = ones(self.D, self.N); W(self.Omegac) = tau;
    CI = C; CI(1:(self.N+1):end) = -1;
    relthr = infnorm(Y(self.Omega));
    % ith slice of repC contains 1 c_i^T.
    % Used for multiplying slices of DxNxN tensor by diag(c_i).
    repCI = repmat(CI, [1 1 self.D]);
    repCI = permute(repCI, [3 1 2]);
    sumCIsqr = sum(CI.^2,2);

    % Pre-compute Cholesky factorizations for Y updates.
    RR = cell(1,self.D);
    for ii=1:self.D
      omegaic = self.Omegac(ii,:);
      A = ldiagmult(W(ii,:)', CI');
      M = (self.lambda*(A(:,omegaic)'*A(:,omegaic)) + ...
          params.mu*diag(sumCIsqr(omegaic)));
      RR{ii} = chol(M);
    end

    % Initialize variables.
    [LL, ULL] = deal(zeros(self.D, self.N, self.N));
    Lrnks = zeros(self.N, 1);
    Y(self.Omega) = self.X(self.Omega);
    repY = repmat(Y, [1 1 self.N]);
    allYdiagCI = repY.*repCI;

    prtformstr = 'k=%d, feasLL=%.2e, minr=%d, maxr=%d, rt=%.2f \n';

    history.status = 1;
    for kk=1:params.maxIter
      itertstart = tic;
      % Update each of the N L^i by SVD shrinkage-thresholding (expensive!!).
      QQ = allYdiagCI - ULL;
      for ii=1:self.N
        Q = QQ(:,:,ii);
        Qnorm = sqrt(sum(Q.^2));
        nzmask = Qnorm > 1e-5*max(Qnorm); % Often entire columns will be zero.
        [LL(:,nzmask,ii), Lrnks(ii)] = prox_nuc(Q(:,nzmask), 1/params.mu);
        % [LL(:,nzmask,ii), Lrnks(ii)] = prox_nuc(Q(:,nzmask), 1/params.mu, Lrnks(ii));
      end
      % Update Y by solving row-by-row least-squares problem.
      for ii=1:self.D
        omegai = self.Omega(ii,:); omegaic = self.Omegac(ii,:);
        xi = self.X(ii,:)';
        A = ldiagmult(W(ii,:)', CI');
        D = squeeze(LL(ii,:,:) + ULL(ii,:,:)); % N x N row-slice of LL+ULL.
        b = -self.lambda*A(:,omegaic)'*(A(:,omegai)*xi(omegai)) + ...
            params.mu*sum(CI(omegaic,:).*D(omegaic,:),2);
        Y(ii,omegaic) = RR{ii} \ (RR{ii}' \ b);
      end
      % Update scaled Lagrange multipliers.
      repY = repmat(Y, [1 1 self.N]);
      allYdiagCI = repY.*repCI;
      conLL = LL - allYdiagCI;
      ULL = ULL + conLL;

      % Diagnostic measures, printing, logging.
      feas = infnorm(conLL)/relthr;
      if params.prtLevel > 0
        fprintf(prtformstr, kk, feas, min(Lrnks), max(Lrnks), toc(itertstart));
      end
      if params.logLevel > 0
        history.feas(kk) = feas;
        history.Lrnk(kk,:) = [min(Lrnks) median(Lrnks) max(Lrnks)];
      end

      if feas < params.convThr
        history.status = 0;
        break
      end
    end
    % Ensure P_Omega constraint is satisfied exactly.
    history.iter = kk;
    history.rtime = toc(tstart);
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
    W = ones(self.D, self.N); W(self.Omegac) = tau;
    lambdamins = zeros(N,1);
    for ii=1:self.N
      wY = ldiagmult(W(:,ii), Y);
      wy = wY(:,ii); wYtrim = trimmat(wY,ii);
      beta = wy'*wYtrim;
      cvx_begin
        variable U(self.D, self.N);
        minimize(norm(U,2));
        subject to
          sum(U.*Y) == beta;
      cvx_end
      lambdamins(ii) = 1/norm(U,2);
    end
    lambdamin = max(lambdamins);
    lambda = alpha*lambdamin;
    end
  end

end
