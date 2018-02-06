function [err, groups, A, rtime, obj, C] = myEnSC(X, true_groups, lamb, gamma, Nsample)
% myEnSC    Wrap Chong's EnSC method for easier evaluation. Solves formulation:
%
%   min_{c_j} lamb/2 || x_j - X c_j||_2^2 + (gamma ||c_j||_1 + (1-gamma)/2
%     ||c_j||_2^2), s.t. c_{jj} = 0.
%
%   [err, groups, A, rtime, obj, C] = myEnSC(X, true_groups, [lamb, gamma, ...
%     Nsample])
%
%   Args:
%     X: D x N data matrix.
%     true_groups: N x 1 group assignment.
%     lamb: Regularization strength [default: .05].
%     gamma: Tradeoff between l_1 and l_2 regularization terms [default: 0.9].
%     Nsample: Maximum size of active set. Recommended to be set around
%       (3~10)*(sparsity of c) [default: 100].
%
%   Returns:
%     err: Cluster error.
%     groups: N x 1 learned cluster assignment.
%     A: N x N affinity.
%     rtime: runtime in seconds for computing sparse-representations.
%     obj: average EN objective.
%     C: N x N sparse representation matrix.

% Set defaults.
if nargin < 3; lamb = 20; end
if nargin < 4; gamma = 0.9; end
if nargin < 5; Nsample = 100; end

% Convert to Chong's notation.
nu0 = lamb; lambda = gamma;

% Normalize columns.
Xnorms = sqrt(sum(X.^2));
X = X ./ repmat(Xnorms+eps, [size(X,1), 1]);

tic;
% ORGEN to compute sparse representations.
EN_solver =  @(X, y, lambda, nu) rfss( X, y, lambda / nu, (1-lambda) / nu );
% EN_solver = @(X, y, lambda, nu) CompSens_EN_Homotopy(X, y, lambda, nu);
C = ORGEN_mat_func(X, EN_solver, 'nu0', nu0, 'nu_method', 'fixed', ...
    'lambda', lambda, 'Nsample', Nsample, 'maxiter', 2, 'outflag', false);
rtime = toc;

obj = mean(lamb*0.5*sum((X-X*C).^2)+(gamma*sum(abs(C))+...
        0.5*(1-gamma)*sum(C.^2)));

% Post-process affinity.
N = length(true_groups);
if max(abs(C(:))) <= 1e-9
  err = 1; groups = []; A = [];
else
  A = build_affinity(C);
  % Apply spectral clustering.
  n = length(unique(true_groups)); 
  groups = SpectralClustering(A, n);

  [err, groups] = eval_cluster_error(groups, true_groups);
end
end
