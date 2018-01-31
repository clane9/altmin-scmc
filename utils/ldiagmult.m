function C = ldiagmult(b, A)
% ldiagmult   Multiply: diag(b) A.
%
%   C = ldiagmult(b, A)
C = repmat(b, [1 size(A,2)]).*A;
end
