classdef GSSC_MC2 < SDLSC_MC2
% GSSC_MC    Solver for SDL based combined subspace clustering and completion,
%   with group-sparsity regularization. Solves formulation
%
%   min_{D,C} lambda/2 ||P_Omega(X - UC)||_F^2 ...
%       + eta_1 ||U||_{2,1} + \sum_{i,j}^{n,N} ||c_{i,j}||_2
%   s.t. ||U||_F <= 1 (only if eta_1 = 0)
%
%   where U = [U_1 ... U_n], U_i in \RR^{D x d}, c_{i,j} in \RR^d ith block of
%   jth column.
%
%   solver = GSSC_MC(X, Omega, n, lambda, eta1, d)

  properties
    d;
  end

  methods

    function self = GSSC_MC2(X, Omega, n, lambda, eta1, d)
    % GSSC_MC    Solver for SDL based combined subspace clustering and completion,
    %   with group-sparsity regularization. Solves formulation
    %
    %   min_{D,C} lambda/2 ||P_Omega(X - UC)||_F^2 ...
    %       + eta_1 ||U||_{2,1} + \sum_{i,j}^{n,N} ||c_{i,j}||_2
    %   s.t. ||U||_F <= 1 (only if eta_1 = 0)
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
    %     eta1: group sparsity penalty parameter for U. If 0, U will be
    %       constrained [default: 0].
    %     d: size of each group dictionary block [default: 0.2*N/n].
    %
    %   Returns:
    %     self: GSSC_MC solver instance.
    if nargin < 5; eta1 = 0; end
    if nargin < 6; d = ceil(0.2*size(X,2)/n); end
    self = self@SDLSC_MC2(X, Omega, n, lambda, eta1, d*n);
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
    if self.eta1 > 0
      Unorms = sqrt(sum(U.^2));
      R = self.eta1*sum(Unorms); % ||U||_{2,1}
    else
      if norm(U, 'fro') > 1; R = inf; else R = 0; end % ||U||_F <= 1
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
