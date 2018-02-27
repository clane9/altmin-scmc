classdef GSSC_MC < SDLSC_MC
% GSSC_MC    Solver for SDL based combined subspace clustering and completion,
%   with group-sparsity regularization. Solves formulation
%
%   min_{D,C} lambda/2 ||P_Omega(X - UC)||_F^2 ...
%       + eta_1 ||U||_{2,1} + \sum_{i,j}^{n,N} ||c_{i,j}||_2
%   s.t. ||U_i||_2 <= 1 for all i.
%
%   where U = [U_1 ... U_n], U_i in \RR^{D x d}, c_{i,j} in \RR^d ith block of
%   jth column.
%
%   solver = GSSC_MC(X, Omega, n, lambda, Uconstrain, eta1, d)

  properties
    d; Uconstrain;
  end

  methods

    function self = GSSC_MC(X, Omega, n, lambda, Uconstrain, eta1, d)
    % GSSC_MC    Solver for SDL based combined subspace clustering and completion,
    %   with group-sparsity regularization. Solves formulation
    %
    %   min_{D,C} lambda/2 ||P_Omega(X - UC)||_F^2 ...
    %       + eta_1 ||U||_{2,1} + \sum_{i,j}^{n,N} ||c_{i,j}||_2
    %   s.t. ||U_i||_2 <= 1 for all i.
    %
    %   where U = [U_1 ... U_n], U_i in \RR^{D x d}, c_{i,j} in \RR^d ith block of
    %   jth column.
    %
    %   solver = GSSC_MC(X, Omega, n, lambda, Uconstrain, eta1, d)
    %
    %   Args:
    %     X: D x N incomplete data matrix.
    %     Omega: D x N binary pattern of missing entries.
    %     n: number of clusters.
    %     lambda: self-expression penalty parameter.
    %     Uconstrain: whether to impose bound constraint on U [default: true].
    %     eta1: group sparsity penalty parameter for U [default: 1].
    %     d: size of each group dictionary block [default: 0.2*N/n].
    %
    %   Returns:
    %     self: GSSC_MC solver instance.
    if nargin < 5; Uconstrain = true; end
    if nargin < 6; eta1 = 1; end
    if nargin < 7; d = ceil(0.2*size(X,2)/n); end
    self = self@SDLSC_MC(X, Omega, n, lambda, eta1, 0, d*n);
    self.Uconstrain = Uconstrain;
    self.d = d;
    end


    function [obj, L, R] = objective(self, U, C)
    % objective   Evaluate objective, loss, and regularization.
    %
    %   [obj, L, R] = solver.objective(U, C)
    %
    %   Args:
    %     U: D x K dictionary.
    %     C: K x N sparse representation coefficient C.
    %
    %   Returns:
    %     obj: Objective value.
    %     L: Reconstruction loss.
    %     R: regularizer.
    Res = self.X - U*C;
    L = 0.5*sum(Res(self.Omega).^2);
    Unorms = sqrt(sum(U.^2));
    R = self.eta1*sum(Unorms); % ||U||_{2,1}
    if self.Uconstrain && max(Unorms) > 1
      R = R + inf; % ||U_i||_2 <= 1 constraint
    end
    % Reshape C as [C_1 ... C_n] where C_i is d x N to compute group sparsity norm.
    Cblocks = reshape(C, self.d, []);
    R = R + sum(sqrt(sum(Cblocks.^2))); % \sum_{i,j} ||c_{i, j}||_2
    obj = self.lambda*L + R;
    end


    function [C, history] = exprC(self, U, C0, params)
    % exprC   Compute sparse representation coefficients by accelerated proximal
    %   gradient.  Solves the formulation
    %
    %   min_C \lambda/2 ||W \odot (X - UC)||_F^2 + ...
    %     \sum_{i,j} ||c_{i,j}||_2
    %
    %   [C, history] = solver.exprC(U, C0, params)
    %
    %   Args:
    %     U: D x K dictionary.
    %     C0: K x N self-expressive coefficient initial guess.
    %     params: optimization parameters for APG.
    %       maxIter: [default: 500].
    %       convThr: [default: 1e-4].
    %       prtLevel: 1=basic per-iteration output [default: 0].
    %       logLevel: 0=basic summary info, 1=detailed per-iteration info
    %         [default: 0]
    %
    %   Returns:
    %     C: K x N self-expression.
    %     history: Struct containing diagnostic info.

    % Set defaults.
    fields = {'maxIter', 'convThr', 'prtLevel', 'logLevel'};
    defaults = {500, 1e-4, 0, 0};
    for i=1:length(fields)
      if ~isfield(params, fields{i})
        params.(fields{i}) = defaults{i};
      end
    end

    W = double(self.Omega); Ut = U';
    function [f, G] = exprC_ffun(C)
    Res = W.*(U*C - self.X);
    f = self.lambda*0.5*sum(sum(Res.^2));
    if nargout > 1
      G = self.lambda*(Ut*(W.*Res));
    end
    end

    function [r, Z] = exprC_rfun(C, rho)
    Cblocks = reshape(C, self.d, []);
    r = sum(sqrt(sum(Cblocks.^2)));
    if nargout > 1
      Zblocks = prox_L21(Cblocks, rho);
      Z = reshape(Zblocks, size(C));
    end
    end

    [C, history] = apg(C0, @exprC_ffun, @exprC_rfun, params);
    end


    function [U, history] = solveU(self, U0, C, params)
    % solveU   Update dictionary by accelerated proximal gradient. Solves the
    %   formulation
    %
    %   min_U \lambda/2 ||W \odot (X - UC)||_F^2 + eta_1 ||U||_{2,1}
    %   s.t.  ||U_i||_2 <= 1 for all i
    %
    %   [U, history] = solver.solveU(U0, C, params)
    %
    %   Args:
    %     U0: D x K dictionary initial guess.
    %     C: K x N self-expressive coefficient.
    %     params: optimization parameters for APG.
    %       maxIter: [default: 500].
    %       convThr: [default: 1e-4].
    %       prtLevel: 1=basic per-iteration output [default: 0].
    %       logLevel: 0=basic summary info, 1=detailed per-iteration info
    %         [default: 0]
    %
    %   Returns:
    %     U: D x K self-expression.
    %     history: Struct containing diagnostic info.

    % Set defaults.
    fields = {'maxIter', 'convThr', 'prtLevel', 'logLevel'};
    defaults = {500, 1e-4, 0, 0};
    for i=1:length(fields)
      if ~isfield(params, fields{i})
        params.(fields{i}) = defaults{i};
      end
    end

    W = double(self.Omega); Ct = C';
    function [f, G] = solveU_ffun(U)
    Res = W.*(U*C - self.X);
    f = self.lambda*0.5*sum(sum(Res.^2));
    if nargout > 1
      G = self.lambda*((W.*Res)*Ct);
    end
    end

    function [r, Z] = solveU_rfun(U, rho)
    r = 0;
    Unorms = sqrt(sum(U.^2));
    if self.Uconstrain && max(Unorms) > 1
      r = r + inf; % ||U_i||_2 <= 1 constraint
    end
    if self.eta1 > 0
      r = r + self.eta1*sum(sqrt(sum(U.^2)));
    end
    if nargout > 1
      Z = U;
      if self.eta1 > 0
        Z = prox_L21(Z, self.eta1*rho);
      end
      Znorms = sqrt(sum(Z.^2));
      % Need to add a little to column norms to avoid numerical constraint failure.
      coeffs = min(1./(Znorms+10*eps), 1);
      if self.Uconstrain
        Z = Z.*repmat(coeffs, [self.D 1]);
      end
    end
    end

    [U, history] = apg(U0, @solveU_ffun, @solveU_rfun, params);
    end


    function [groups, A, cluster_err] = cluster(self, C, groupsTrue)
    % cluster   Cluster by assigning each data point to it's largest block.
    %
    %   [groups, A, cluster_err] = solver.cluster(C, groupsTrue)
    CC = reshape(C, [self.d, self.n, self.N]);
    seg = squeeze(sum(CC.^2,1));
    [~, groups] = max(seg, [], 1); groups = groups';
    if nargin > 2
      [cluster_err, groups] = eval_cluster_error(groups, groupsTrue);
    else
      cluster_err = nan;
    end
    A = [];
    end

  end
end
