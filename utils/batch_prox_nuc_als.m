function [UU, VV] = batch_prox_nuc_als(UU, VV, ZZ, lamb)
% batch_prox_nuc_als    Solve nuclear norm proximal operator for a batch of
%   matrices ZZ, by alternating least squares.
%
%   min_{U_i,V_i} lamb/2 (||U_i||_F^2 + ||V_i||_F^2) ...
%     + 1/2 ||U_i V_i^T - Z_i||_F^2
%
%   [UU, VV] = batch_prox_nuc_als(UU, VV, ZZ, lamb)

[~, N, M] = size(ZZ); d = size(UU,2);
lambbigI = lamb*speye(d*M);

% UUstack = reshape(UU, size(UU,1), []);
VVstack = reshape(VV, size(VV,1), []);
ZZstack = reshape(ZZ, size(ZZ,1), []);
ZZt = permute(ZZ, [2 1 3]);
ZZtstack = reshape(ZZt, size(ZZt,1), []);

for ii=1:20
  % Even though these are very sparse block diagonal matrices, these
  % multiplications are still slow... Must be about how the entries are stored.
  % Computing standard proximal operator in a loop is still faster.
  VVbd = stacktobd(VVstack, d);
  AU = (lambbigI + (VVbd'*VVbd));
  bU = ZZstack*VVbd;
  UUstack = bU / AU;
  UUbd = stacktobd(UUstack, d);
  AV = (lambbigI + (UUbd'*UUbd));
  bV = ZZtstack*UUbd;
  VVstack = bV / AV;
  obj = lamb*0.5*(sum(UUstack(:).^2) + sum(VVstack(:).^2)) + ...
     0.5*sum(sum((UUbd*VVbd' - stacktobd(ZZstack, N)).^2));
  fprintf('%.4e \n', obj);
end
UU = reshape(UUstack, size(UU));
VV = reshape(VVstack, size(VV));
end

function QQbd = stacktobd(QQ, d)
QQcell = mat2cell(sparse(QQ), size(QQ,1), d*ones(size(QQ,2)/d,1));
QQbd = blkdiag(QQcell{:});
end
