classdef GSSC_MC_l0 < GSSC_MC
% GSSC_MC_l0    Solver for SDL based combined subspace clustering and completion,
%   with l0 group-sparsity regularization. Solves formulation
%
%   min_{D,C} lambda/2 ||P_Omega(X - UC)||_F^2 ...
%       + eta_1 ||U||_{2,1}
%   s.t. (#_{i=1}^n) ||c_{i,j}||_2 > 0) <= 1 for all j=1,..,N
%        ||U_i||_2 <= 1 for all i=1,..,D.
%
%   where U = [U_1 ... U_n], U_i in \RR^{D x d}, c_{i,j} in \RR^d ith block of
%   jth column.
%
%   solver = GSSC_MC_l0(X, Omega, n, lambda, Uconstrain, eta1, d)


  methods

    function self = GSSC_MC_l0(X, Omega, n, lambda, Uconstrain, eta1, d)
    % GSSC_MC_l0    Solver for SDL based combined subspace clustering and completion,
    %   with l0 group-sparsity regularization. Solves formulation
    %
    %   min_{D,C} lambda/2 ||P_Omega(X - UC)||_F^2 ...
    %       + eta_1 ||U||_{2,1}
    %   s.t. (#_{i=1}^n) ||c_{i,j}||_2 > 0) <= 1 for all j=1,..,N
    %        ||U_i||_2 <= 1 for all i=1,..,D.
    %
    %   where U = [U_1 ... U_n], U_i in \RR^{D x d}, c_{i,j} in \RR^d ith block of
    %   jth column.
    %
    %   solver = GSSC_MC_l0(X, Omega, n, lambda, Uconstrain, eta1, d)
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
    self = self@GSSC_MC(X, Omega, n, lambda, Uconstrain, eta1, d);
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
    % Reshape C as a d x n x N tensor to evaluate l0 sparsity.
    CC = reshape(C, [self.d, self.n, self.N]);
    seg = squeeze(sum(CC.^2,1));
    % Check if any data points register non-zero coeff in > 1 C block.
    if any(sum(seg > 0) > 1)
      R = R + inf;
    end
    obj = self.lambda*L + R;
    end


    function [C, history] = exprC(self, U, C0, params)
    % exprC   Compute sparse representation coefficients by accelerated proximal
    %   gradient.  Solves the formulation
    %
    %   min_C \lambda/2 ||W \odot (X - UC)||_F^2 + ...
    %   s.t. (#_{i=1}^n) ||c_{i,j}||_2 > 0) <= 1 for all j=1,..,N
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

    function [r, Z] = exprC_rfun(C, ~)
    r = 0;
    CC = reshape(C, [self.d, self.n, self.N]);
    seg = squeeze(sum(CC.^2,1));
    % Check if any data points register non-zero coeff in > 1 C block.
    if any(sum(seg > 0) > 1)
      r = r + inf;
    end
    if nargout > 1
      [~, maxInd] = max(seg, [], 1);
      % stride index of maximum group into 3rd dimension of CC.
      strdMaxInd = maxInd + self.n*(0:(self.N-1));
      % Pull out maximum group into Z.
      Cblocks = reshape(C, self.d, []);
      Zblocks = zeros(size(Cblocks));
      Zblocks(:, strdMaxInd) = Cblocks(:, strdMaxInd);
      Z = reshape(Zblocks, size(C));
    end
    end

    [C, history] = apg(C0, @exprC_ffun, @exprC_rfun, params);
    end


  end
end
