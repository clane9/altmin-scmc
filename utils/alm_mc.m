function [L, history] = alm_mc(M, Omega, params)
% alm_mc    Inexact ALM algorithm for matrix completion from Lin et al 2009.
%   Solves formulation
%
%   min_L ||L||_* s.t. L = M + E, P_Omega(E) = 0
%
%   [L, history] = alm_mc(M, Omega, params)
%
%   Args:
%     M: D x N incomplete data matrix.
%     Omega: D x N logical indicator of observed entries.
%     params: Struct containing problem parameters.
%       mu: ALM penalty parameter [default: 1/||M||_{2,1}].
%       alpha: rate for increading mu after each iteration [default: 1.1].
%       maxIter: [default: 100].
%       convThr: [default: 1e-3].
%       prtLevel: 1=basic per-iteration output [default: 0].
%       logLevel: 0=basic summary info, 1=detailed per-iteration info
%         [default: 0]
%
%   Returns:
%     L: D x N low-rank completion.
%     history: Struct containing diagnostic info.
if nargin < 3; params = struct; end
% Set defaults.
% see page 6 in https://pdfs.semanticscholar.org/1556/6adc0e6312b72290c31f891bdb080c06d997.pdf
% for mu/alpha default justification.
fields = {'mu', 'alpha', 'maxIter', 'convThr', 'prtLevel', 'logLevel'};
defaults = {1/sum(sqrt( sum(M .^2, 1) )), 1.1, 100, 1e-3, 0, 0};
for i=1:length(fields)
  if ~isfield(params, fields{i})
    params.(fields{i}) = defaults{i};
  end
end
mu = params.mu; alpha = params.alpha;
tstart = tic; % start timer.

Omega = logical(Omega); Omegac = ~Omega;
M(Omegac) = 0; % zero-fill missing entries.
[D, N] = size(M);
L = zeros(D, N);
E = zeros(D, N);
U = zeros(D, N);

relthr = infnorm(M);
history.status = 1;
for kk=1:params.maxIter
  L = prox_nuc(M - E + U, 1/mu);
  E = Omegac.*(M-L + U);
  U = U + (M - E - L);
  feas = infnorm(M - E - L)/ relthr;
  if params.prtLevel > 0
    fprintf('k=%d, feas=%.2e \n', kk, feas);
  end
  if params.logLevel > 0
    history.feas(kk) = feas;
  end
  if feas < params.convThr
    history.status = 0;
    break
  end
  mu = alpha*mu;
end
history.iter = kk; history.rtime = toc(tstart);
end
