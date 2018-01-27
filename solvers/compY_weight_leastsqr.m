function Y = compY_weight_leastsqr(X, Omega, C, W)
% compY_weight_leastsqr   Complete missing data using self-expression C. Solves
%   objective:
%
%   min_Y 1/2 ||W \odot (Y - YC)||_F^2 s.t. P_Omega(Y-X) = 0.
%
%   Problem is solved row-by-row by computing a least-squares solution using SVD.
%
%   Args:
%     X: D x N incomplete data matrix.
%     Omega: D x N binary pattern of missing entries.
%     C: N x N self-expressive coefficient C.
%     W: Non-negative weight matrix
%
%   Returns:
%     Y: D x N completed data.
Omega = logical(Omega);
[D, N] = size(X);
IC = eye(N) - C;
Y = X; % Initialize Y so that it agrees on observed entries.
for ii=1:D
  omegai = Omega(ii,:); omegaic = ~omegai;
  xi = X(ii,:)'; wi = W(ii,:)';
  % Compute A = ((I - C) diag(W_{i,.}))^T. Drop rows of A set to zero.
  A = IC' .* repmat(wi, [1 N]); A = A(wi~=0, :);
  % Compute least squares solution to:
  %   1/2 ||A_{\omega_i^c} y_{\omega_i^c} + A_{\omega_i} x_{\omega_i}||_2^2
  Y(ii,omegaic) = pinv(A(:,omegaic))*(-A(:,omegai)*xi(omegai));
end
end
