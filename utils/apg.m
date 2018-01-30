function [x, history] = apg(x0, ffun, rfun, params)
% apg    Generic accelerated proximal gradient descent algorithm
%
%   [x, history] = apg(x0, ffun, rfun, params)
%
%   Minimize an objective f(x) + r(x) using the accelerated proximal gradient
%   descent method.
%
%   Args:
%     x0: Initial guess.
%     ffun, rfun: string function names or function handles.
%       [f, g] = ffun(x) returns function value and optionally gradient.
%       r = rfun(x) returns value of r at x.
%       [r, z] = rfun(u, alpha) returns r value and proximal step from u, with
%         penalty alpha.
%     params: Struct containing parameters for optimization:
%       maxIter: Maximum iterations [default: 500].
%       convThr: Stopping tolerance [default: 1e-6].
%       prtLevel: 1=basic per-iteration output [default: 0].
%       logLevel: 1=basic summary info, 2=detailed per-iteration info
%         [default: 1]
%
%   Returns:
%     x: final iterate.
%     history: Struct containing containing fields:
%       iter: Total iterations.
%       status: 0 if stopping tolerance achieved, 1 otherwise.
%       obj, f, r: objective, f value, r value (per-iteration).
%       update: Relative inf norm change in iterates (per-iteration).
%       alpha: Minimum step size (per-iteration).

if nargin < 4
  params = struct;
end
fields = {'maxIter', 'convThr', 'prtLevel', 'logLevel'};
defaults = {500, 1e-6, 0, 1};
for i=1:length(fields)
  if ~isfield(params, fields{i})
    params.(fields{i}) = defaults{i};
  end
end
tic; % start timer.

% If f, r functions are strings, convert to function handles.
if ischar(ffun)
  ffun = str2func(ffun);
end
if ischar(rfun)
  rfun = str2func(rfun);
end

% Initialize step size, iterates
alpha = 1; alphamin = 1e-7; eta = 0.5;
x = x0; xprev = x0;
relthr = max(1, infnorm(x0));

% Print form str.
printformstr = 'k=%d \t update=%.2e \t obj=%.2e \t f=%.2e \t r=%.2e \t alpha=%.2e \n';

% Accelerated proximal gradient loop.
status = 1; iter = 0;
while iter <= params.maxIter
  iter = iter + 1;
  % Acceleration (Proximal Algorithms, Sect. 4.3)
  beta = (iter-1) / (iter + 2);
  y = x + beta*(x - xprev);
  xprev = x;

  % Compute f value, gradient.
  [f, g] = ffun(y);

  % Determine step size alpha < 2/L using the simple backtracking strategy.
  while 1
    [~, x] = rfun(y - alpha*g, alpha);
    step = x - y;
    fhat = f + frodot(g, step) + (0.5/alpha)*fronormsqrd(step);
    if ffun(x) <= fhat || alpha <= alphamin
      break
    else
      alpha = eta*alpha;
    end
  end

  f = ffun(x); r = rfun(x); obj = f + r;
  update = infnorm(x - xprev) / relthr;

  if params.prtLevel > 0
    fprintf(printformstr, iter, update, obj, f, r, alpha);
  end

  if params.logLevel > 1
    history.obj(iter) = obj;
    history.f(iter) = f;
    history.r(iter) = r;
    history.update(iter) = update;
    history.alpha(iter) = alpha;
  end

  % Check stopping tolerance.
  if update < params.convThr
    status = 0;
    break
  end
end
history.iter = iter; history.status = status; history.rtime = toc;
end

function t = frodot(x, y)
t = sum(x(:).*y(:));
end

function t = fronormsqrd(x)
t = sum(x(:).^2);
end

function t = infnorm(x)
t = max(abs(x(:)));
end
