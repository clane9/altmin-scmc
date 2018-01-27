function A = build_affinity(C, normalize, thresh)
% build_affinity    Convert self-expression matrix C to symmetric affinity.
%
%   A = build_affinity(C, normalize, thresh)
%
%   Args:
%     C: N x N self-expression.
%     normalize: whether to normalize each column to have unit max abs entry
%       [default: false].
%     thresh: whether to threshold entries (<1e-8) and sparsify 
%       [default: true].
%
%   Returns:
%     A: Affinity matrix.

if nargin < 2; normalize = false; end
if nargin < 3; thresh = true; end

N = size(C, 1);
absC = abs(C);
absC(1:(N+1):end) = 0; % set diagonal to zero.
if normalize
  % normalize each column to have max entry 1.
  absC = absC ./ repmat(max(absC) + eps, [N 1]);
end
if thresh
  % threshold small values and convert to sparse matrix.
  Cabs(Cabs < 1e-8) = 0; Cabs = sparse(Cabs);
end
A = absC + absC';
end
