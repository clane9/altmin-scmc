function [X, groupsTrue, Omega] = generate_scmd_data_matrix(n, d, D, Ng, ...
    sigma, rho, delta, seed)
% generate_scmd_data_matrix   Generate a synthetic dataset containing several
%   subspaces with missing data.
%
%   [X, Omega, groupsTrue] = generate_scmd_data_matrix(n, d, D, Ng, sigma, ...
%     rho, delta, seed)
%
%   Complete data are sampled as
%
%   X = [U_1 V_1^T ... U_n V_n^T] + E
%
%   where the U_i are random orthonormal bases of dimension D x d, V_i are
%   random N_g x d coefficients sampled from N(0,1), and E is D x N_g*n dense
%   Gaussian noise sampled from N(0,sigma). The pattern of missing entries is
%   constructed by first uniformly selecting delta fraction of columns to
%   corrupt, and then dropping (on average) rho fraction of entries uniformly
%   from each.
%
%   Args:
%     n: Number of subspaces.
%     d: Subspace dimension.
%     D: Ambient dimension.
%     Ng: Number of points per group.
%     sigma: Dense noise sigma.
%     rho: Fraction of missing entries per corrupted column.
%     delta: Fraction of corrupted columns.
%     seed: Seed for random generator.
%
%   Returns:
%     X: D x (Ng*n) complete, noisy data matrix.
%     groupsTrue: (Ng*n) x 1 cluster assignment.
%     Omega: D x (Ng*n) logical indicator matrix of observed entries, so that the
%       observed data Xobs = X.*Omega.
rng(seed);

% Sample noiseless data from n subspaces.
N = Ng*n;
Xs = cell(1, n);
groupsTrue = zeros(N, 1);
for ii=1:n
  U = orth(randn(D, d));
  V = randn(d, Ng);
  Vnorm = sqrt(sum(V.^2)); V = V ./ repmat(Vnorm, [d 1]);
  % Scaling by sqrt(D) so that norm of X_i is sqrt(D).
  % This way noise sigma has same impact across D, d.
  Xs{ii} = sqrt(D)*U*V;
  startidx = (ii-1)*Ng + 1; stopidx = startidx + Ng - 1;
  groupsTrue(startidx:stopidx) = ii;
end
X = [Xs{:}];

% Add dense noise.
if sigma > 0
  X = X + randn(size(X))*sigma;
end

% Generate pattern of missing entries by generating a sub-matrix indicating
% corrupted columns.
if min([rho delta]) > 0
  K = round(delta*N); % number corrupted columns.
  M = round(rho*D*K); % number unobserved entries.
  unobsInd = randperm(D*K, M);
  Omega = ones(D, K); Omega(unobsInd) = 0;
  Omega = [Omega ones(D, N-K)];
  Omega = Omega(:, randperm(N));
  Omega = logical(Omega);
else
  Omega = true(D, N);
end
end
