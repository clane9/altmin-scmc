function [C, history] = exprC_weight_ensc_admm(Y, Omega, C, tau, params)

% Set defaults.
fields = {'lambda', 'gamma', 'mu', 'maxIter', 'convThr', ...
    'prtLevel', 'logLevel'};
defaults = {10, 0.9, 10, 200, 1e-6, 1, 1};
for i=1:length(fields)
  if ~isfield(params, fields{i})
    params.(fields{i}) = defaults{i};
  end
end

% Set constants.
[D, N] = size(Y);
tausqr = tau^2;
Omega = logical(Omega); Omegac = ~Omega;
W = ones(D, N); W(Omegac) = tau;

YtY = Y'*Y;
R = chol(params.lambda*(tausqr+1)*YtY + params.mu*eye(N));
relthr = infnorm(Y(Omega));

% Initialize variables.
Res = Y*C - Y; A = Omegac .* Res; B = Omega .* Res;
U = zeros(N); % scaled Lagrange multiplier

prtformstr = 'k=%d, obj=%.2e, feas=%.2e \n';

history.status = 1;
for kk=1:params.maxIter
  % Update Z (proxy for C) by least squares.
  leastsqr_target = params.lambda*YtY + params.mu*(C - U);
  if tau < 1
    leastsqr_target = leastsqr_target + params.lambda*Y'*A;
    if tau > 0
      leastsqr_target = leastsqr_target + params.lambda*tausqr*(YtY + Y'*B);
    end
  end
  Z = R \ (R' \ leastsqr_target);
  % Update C by solving elastic-net proximal operator, with diagonal constraint.
  C = prox_en(Z + U, params.gamma, 1/params.mu);
  C(1:(N+1):end) = 0; % set diagonal to 0.
  % Update variables used to absorb errors on \Omega, \Omega^c.
  Res = Y*Z - Y;
  if tau < 1
    A = Omegac .* Res;
    if tau > 0
      B = Omega .* Res;
    end
  end
  % Update scaled Lagrange multiplier.
  U = U + (Z - C);
  
  % Diagnostic measures, printing, logging.
  feas = infnorm(Z - C)/relthr;
  obj = (params.lambda/2)*sum(sum((W.*Res).^2)) + ...
      params.gamma*sum(abs(C(:))) + ((1-params.gamma)/2)*sum(C(:).^2);
  if params.prtLevel > 0
    fprintf(prtformstr, kk, obj, feas);
  end
  if params.logLevel > 0
    history.obj(kk) = obj;
    history.feas(kk) = feas;
  end

  if feas < params.convThr
    history.status = 0;
    break
  end
end
history.iter = kk;
end
