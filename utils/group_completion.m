function Y = group_completion(X, Omega, groups, n)
% group_completion    Complete a partially observed X separately for each group
%   of columns.
%
%   Y = group_completion(X, Omega, groups, n)
%
%   Args:
%     X: D x N incomplete data matrix.
%     Omega: D x N indicator matrix of observed entries.
%     groups: N x 1 cluster assignment.
%     n: number of groups.
%
%   Returns:
%     Y: D x N completed data matrix.

X = X.*Omega;
Y = zeros(size(X));

for i=1:n
  indices = find(groups == i);
  Y(:, indices) = alm_mc(X(:, indices), Omega(:, indices));
end
end
