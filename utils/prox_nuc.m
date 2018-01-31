function [A, r] = prox_nuc(X, lamb, r0, incre)
% prox_nuc    Evaluate proximal operator of nuclear norm.
%
%   A = prox_nuc(X, lamb, r0)
%
%   Args:
%     X: D x N matrix.
%     lamb: Singular value threshold.
%     r0: Guess of threshold rank. If provided and positive, will use svds to
%       compute only a few singular vectors, incrementing by incre until
%       threshold rank is found (Follows SVT.m written by Candes et al).
%     incre: How much to increment rank by if smallest singular value exceeds
%       threshold [default: max(5, ceil(.05*min(size(X))))].
%
%   Returns:
%     A: D x N thresholded matrix.
%     r: Rank of A.

% Short-circuit option for trivial case.
mindim = min(size(X));
if mindim == 1
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
  if nargin < 4; incre = max(5, ceil(.05*mindim)); end
  OK = 0; r = r0;
  while ~OK
    r = min(r+incre, mindim);
    [U, S, V] = svds(X, r);
    OK = (S(end) <= lamb) || ( r == mindim );
  end
end
s = diag(S);
r = sum(s > lamb);
sthr = s(1:r) - lamb;
A = U(:,1:r)*diag(sthr)*V(:,1:r)';
end
