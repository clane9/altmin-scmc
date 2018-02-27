function [ groups, C, Y, history ] = PZF_EnSC( X, Omega, n, lambda, varargin )
% PZF_EnSC Projected and Zero Filled Elastic Net Subspace Clustering
% 
%   [groups, C, Y, history] =  PZF_EnSC(X, Omega, n, lambda, mu, maxIter,
%   convThr, gamma) Solves the optimization problem:
%
%   min_C ||C||_1 + \frac{\gamma}{2} ||C||_F^2 + \frac{\lambda}{2} ||P_\Omega(X - XC)||_F^2
% 
%   Which is equivalent to:
% 
%   min_C, A ||C||_1 + \frac{\gamma}{2} ||C||_F^2  + \frac{\lambda}{2} ||X - XC + A||_F^2 s.t. P_\Omega (A) = 0
%   
%   Note that the initial optimization problem may also be solved column
%   by column. When solving for c_j, we project all other points onto the
%   pattern of observed entries of x_j, otherwise denoted as \Omega_j. In
%   this case, we suggest using the Oracle Based Active Set method proposed
%   by You et. al 2016.
% 
%   Args:
%       X: D x N incomplete data matrix.
%       Omega: D x N indicator matrix of observed entries.
%       n: Number of subspaces.
%       lambda: Penalty for reconstruction error
%       mu: Augmented Lagrangian penalty parameter (scalar) [default: 10].
%       maxIter: maximumum iterations [default: 200]
%       convThr: convergence threshold [default: 1e-6]
%       gamma: elastic net parameter [default: 1/9]
%   Returns:
%       groups: N x 1 cluster assignment
%       C: Self-expressive coefficient C
%       Y: D x N completed data.
%       history: struct containing diagnostic info

% Set defaults.
% Default gamma follows (You et al., CVPR 2016).
mu = 10; maxIter = 400; convThr = 1e-6; gamma = 1/9;
if nargin >= 5; mu = varargin{1}; end
if nargin >= 6; maxIter = varargin{2}; end
if nargin >= 7; convThr = varargin{3}; end
if nargin >= 8; gamma = varargin{4}; end
[D, N] = size(X);
normX = norm(X, 'fro');

% Set up initial matrices and parameters
Z = zeros(N);
C = zeros(N);
L = zeros(N);
A = zeros(D, N);

% Speed optimizations
lambdaXtX = lambda * (X' * X);
I = eye(N);
Q = inv(lambdaXtX + mu * I + gamma * I);

k = 1;
convergence = false;
updateHistory = (nargout > 3); % Prevents checking number of arguments in the loop
while (~convergence && k <= maxIter)
    Z_last = Z; A_last = A; C_last = C;
    % Update Z
    Z =  Q*(lambdaXtX + lambda*(X'*A) + mu * (C - L / mu));
    C = max( abs(Z + L/mu) - 1/mu, 0 ) .* sign(Z + L/mu);
    
    % Update C
    C = C - diag(diag(C));
    
    % Update A
    A = (~Omega).*(X*C - X);
    
    % Update L
    L = L + mu * (Z - C);
    
    % Stopping condition.
    res_1 = norm(Z - C, 'fro')/normX; % Primary residual from constraint set.
    convZ = norm(Z - Z_last, 'fro')/normX;
    convA = norm(A - A_last, 'fro')/normX;
    convC = norm(C - C_last, 'fro')/normX;
    stop_cond = max([res_1 convZ convA convC]);

    if updateHistory 
        history.stop_cond(k) = stop_cond;
        history.medCspr(k) = median(1.0 - mean(abs(C) > 1e-3));
        history.reconerr(k) = mean(sum(Omega.*(X - X*C).^2));
    end
    
    convergence = stop_cond < convThr;

    k = k+1;
end

% Spectral clustering.
Csym = build_affinity(C - diag(diag(C)));
groups = spectral_clustering(Csym, n);
Y = group_completion(X, Omega, groups, n);
end
