function grad_error = grad_check(vecfun, X, epsilon)
% grad_check    Check the gradient of a function numerically.
%
%   grad_error = grad_check(fun, X, epsilon) fun is a function handle that
%   takes a vector X as input. It should return the function value, and
%   optionally the gradient. X is a column vector, and epsilon is a step size
%   for computing the numerical gradient.

% First evaluate to get the analytic gradient.
[~, G] = vecfun(X);

numG = zeros(size(X));

% Compute numerical gradient using centered formulation.
% NOTE: What happens if X is big and epsilon is small? We'll get a spurious 0 gradient?
for i=1:numel(X)
  oldval = X(i);
  X(i) = oldval + epsilon;
  fxph = vecfun(X);

  X(i) = oldval - epsilon;
  fxmh = vecfun(X);

  numG(i) = (fxph - fxmh) / (2*epsilon);
  X(i) = oldval;
end

% Return relative error in gradient, using Frobenius norm.
grad_error = norm(G - numG, 'fro') / norm(G, 'fro');
end
