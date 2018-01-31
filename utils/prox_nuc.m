function [A, r] = prox_nuc(X, lamb, r0, incre)
% prox_nuc    Evaluate proximal operator of nuclear norm.
%
%   A = prox_nuc(X, lamb, r0)
%
%   Args:
%     X: D x N matrix.
%     lamb: Singular value threshold.
%     r0: Guess of threshold rank. If provided and positive, will use PROPACK
%       lansvd to compute only a few singular vectors, incrementing by incre
%       until threshold rank is found.
%     incre: How much to increment rank by if smallest singular value exceeds
%       threshold [default: 5].
%
%   Returns:
%     A: D x N thresholded matrix.
%     r: Rank of A.
%
%   Requires PROPACK (http://sun.stanford.edu/~rmunk/PROPACK/) for
%   rank-guessing option.

% Short-circuit option for trivial case.
if min(size(X)) == 1
  s = norm(X);
  r = double(s > lamb);
  sthr = max(s - lamb, 0);
  A = (sthr/s)*X;
  return
end

if nargin < 3 || r0 <= 0
  [U, S, V] = svd(X, 'econ');
else
  % Following SVT.m written by Emmanuel Candes et al.
  if nargin < 4; incre = 5; end
  OK = 0; r = r0; mindim = min(size(X));
  while ~OK
    r = min(r+incre, mindim);
    % Requires PROPACK (http://sun.stanford.edu/~rmunk/PROPACK/)
    [U, S, V] = lansvd(X, r, 'L');
    OK = (S(end) <= lamb) || ( r == mindim );
  end
end
s = diag(S);
r = sum(s > lamb);
sthr = s(1:r) - lamb;
A = U(:,1:r)*diag(sthr)*V(:,1:r)';
end
