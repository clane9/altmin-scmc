function A = prox_nuc(X, lamb)
% prox_nuc    Evaluate proximal operator of nuclear norm.
%
%   A = prox_nuc(X, lamb)
[U, S, V] = svd(X, 'econ');
s = diag(S);
r = sum(s > lamb);
sthr = s(1:r) - lamb;
A = U(:,1:r)*diag(sthr)*V(:,1:r)';
end
