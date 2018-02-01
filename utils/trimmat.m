function Ztrim = trimmat(Z, ind)
% trimmat   Remove index ind from matrix Z.
%
%   Ztrim = trimmat(Z, ind)
Ztrim = [Z(:, 1:(ind-1)) Z(:, (ind+1):end)];
end
