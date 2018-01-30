function L = alm_mc(M, Omega, mu, alpha)
% alm_mc    Inexact ALM algorithm for matrix completion from Lin et al 2009
%
%   L = alm_mc(M, Omega, mu, alpha)
%
%   Args:
%     M: D x N incomplete data matrix.
%     Omega: D x N logical indicator of observed entries.
%     mu, alpha: AlM parameters.
%
%   Returns:
%     L: D x N low-rank completion.

Omega = logical(Omega); Omegac = ~Omega;
[D, N] = size(M);
Y = zeros(D, N);
L = zeros(D, N);
E = zeros(D, N);

if nargin < 4
  mu = 1/sum(sqrt( sum(M .^2, 1) ));
  alpha = 1.1; % see page 6 in https://pdfs.semanticscholar.org/1556/6adc0e6312b72290c31f891bdb080c06d997.pdf
end

% Convergence conditions
k = 1;
maxiter = 100;
relthr = infnorm(M);
convergence = infnorm(M - E - L)/ relthr;
conv_thr = 1e-3; %convergence threshold

while (k < maxiter && convergence > conv_thr)
  L = prox_nuc(M - E + Y/mu, 1/mu);
  E = Omegac.*(M-L + Y/mu);
  Y = Y + mu*(M - E - L);
  convergence = infnorm(M - E - L)/ relthr;
  mu = alpha*mu;
  k = k+1;
end
end
