function A = prox_L21(X, lamb)
% prox_L21    Evaluate proximal operator of \ell_2,1 matrix norm:
%   ||X||_{2,1} = \sum_i ||X_{.,i}||_2
%
%   A = prox_L21(X, lamb)
colnorms = sqrt(sum(X.^2));
[D, N] = size(X);
coeffs = max(ones(1, N) - lamb./colnorms, 0);
A = (ones(D,1)*coeffs).*X;
end
