function [C, Y, history] = SSC_MC(X, Omega, lambda, maxIter, X0)
% \min_C, Y ||C||_1 + lambda/2 ||Y - YC||_F^2
% s.t. P_Omega(Y - X) = 0
% Algorithm:
% - Update C column-wise
% - Update Y row-wise

% parameter
rho = lambda; % for admm

% setup
[D, N] = size(X);

Multiplier = zeros(N); % for admm
B = zeros(D, N); % for admm
C = zeros(N);

% initialize Y by matrix completion
Y = simple_alm_mc( X, Omega );
Y(Omega) = X(Omega);

history.initial_completion_error = norm(Y - X0, 'fro') / norm(X0, 'fro');
history.obj = ones(1, maxIter) * inf;
history.completion_error = ones(1, maxIter) * inf;
for iter = 1:maxIter
    % Update C by ADMM
    YtY = Y' * Y;
    R = chol( lambda*YtY + rho*eye(N) );
    for iter_admm = 1:200
        A = R \ (R' \ ( lambda*YtY + lambda*Y'*B + rho*(C - Multiplier/rho) ) );
        C = max(0, ( abs(A + Multiplier/rho) - 1/rho ) ) .* sign( A + Multiplier/rho );
        C = C - diag(diag(C));
        B = (Y*A - Y) .* (~Omega); % for projected
%        B = zeros(D, N); % otherwise
        Multiplier = Multiplier + rho * (A - C);
        if norm(A - C, 'fro') < 0.001
           break;
        end
    end
    
    % Update Y
    IC = (eye(N) - C);
    for ii = 1:D
        mask = Omega(ii, :);
%             ICi = IC(:, mask); % for projected
        ICi = IC(:, :); % otherwise
        Y(ii, ~mask) = - pinv(ICi(~mask, :)') * (X(ii, mask) * ICi(mask, :))';
%         Y(ii, ~mask) = - pinv([IC(~mask, mask), 5 * IC(~mask, ~mask)]') * ...
%                          (X(ii, mask) * [IC(mask, mask), 5  * IC(mask, ~mask)])';
    end

%     history.obj(iter) = sum(abs(C(:))) + lambda / 2 * norm((Y - Y * C) .* Omega, 'fro') ^2; % projected
    history.obj(iter) = sum(abs(C(:))) + lambda / 2 * norm((Y - Y * C), 'fro') ^2; % otherwise
    history.completion_error(iter) = norm(Y - X0, 'fro') / norm(X0, 'fro');
    
    if iter == 1, obj_old = inf; else, obj_old = history.obj(iter-1); end
    obj_now = history.obj(iter);
    if (obj_now > obj_old || (obj_old - obj_now)/obj_now < 0.005)
        break;
    end
%     if proj == 1 && (obj_now > obj_old || (obj_old - obj_now)/obj_now < 1e-4)
%        history.init_iter = iter;
%        proj = 0;
% %        break;
%     end
end
history.iter = iter;
end

