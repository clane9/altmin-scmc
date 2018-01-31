function C = rdiagmult(A, b);
% rdiagmult   Multiply: A diag(b)
%
%   C = rdiagmult(A, b)
C = A.*repmat(b', [size(A,1) 1]);
end
