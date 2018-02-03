classdef CASS_MC2 < CASS_MC
% CASS_MC2   Solver for CASS regularized alternating minimization for
%   joint subspace clustering and completion. Solves formulation
%
%   min_{Y,C} lambda/2 ||W .* (Y - YC)||_F^2 ...
%       + \sum_{i=1}^N ||Y diag(c_i - e_i)||_*
%   s.t. diag(C) = 0, P_{Omega}(Y - X) = 0
%
%   Solves completion with P_Omega constraint enforced exactly. This comes at
%   some per-iteration cost, but hopefully better convergence/accuracy.
%
%   solver = CASS_MC2(X, Omega, n, lambda)

  properties
  end

  methods

    function self = CASS_MC2(X, Omega, n, lambda)
    % CASS_MC2   Solver for CASS regularized alternating minimization for
    %   joint subspace clustering and completion. Solves formulation
    %
    %   min_{Y,C} lambda/2 ||W .* (Y - YC)||_F^2 ...
    %       + \sum_{i=1}^N ||Y diag(c_i - e_i)||_*
    %   s.t. diag(C) = 0, P_{Omega}(Y - X) = 0
    %
    %   Solves completion with P_Omega constraint enforced exactly. This comes
    %   at some per-iteration cost, but hopefully better convergence/accuracy.
    %
    %   solver = CASS_MC2(X, Omega, n, lambda)
    %
    %   Args:
    %     X: D x N incomplete data matrix.
    %     Omega: D x N binary pattern of missing entries.
    %     n: number of clusters.
    %     lambda: self-expression penalty parameter.
    %
    %   Returns:
    %     self: CASS_MC2 solver instance.
    self = self@CASS_MC(X, Omega, n, lambda);
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
      omegai = self.Omega(ii,:); omegaic = self.Omegac(ii,:);
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

  end

end
