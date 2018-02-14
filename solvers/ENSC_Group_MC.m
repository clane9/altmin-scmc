classdef ENSC_Group_MC < ENSC_MC
% ENSC_Group_MC   Solver for alternating elastic-net subspace clustering (with
%   projection), and group-wise completion.
%
%   solver = ENSC_Group_MC(X, Omega, n, lambda, gamma)


  methods

    function self = ENSC_Group_MC(X, Omega, n, lambda, gamma)
    % ENSC_Group_MC   Solver for alternating elastic-net subspace clustering (with
    %   projection), and group-wise completion.
    %
    %   solver = ENSC_Group_MC(X, Omega, n, lambda, gamma)
    %
    %   Args:
    %     X: D x N incomplete data matrix.
    %     Omega: D x N binary pattern of missing entries.
    %     n: number of clusters.
    %     lambda: self-expression penalty parameter.
    %     gamma: elastic-net tradeoff parameter.
    %
    %   Returns:
    %     self: ENSC_Group_MC solver instance.
    self = self@ENSC_MC(X, Omega, n, lambda, gamma, 0);
    end

    function [Y, history] = compY(self, ~, C, ~, params)
    % compY   Cluster the data using C then complete missing data separately
    %   for each group.
    %
    %   [Y, history] = solver.compY(~, C, ~, ~)
    %
    %   Args:
    %     Y: D x N complete data matrix initial guess (not used, included for
    %       consistency).
    %     C: N x N self-expressive coefficient C.
    %     tau: Non-negative scalar representing reconstruction penalty weight on
    %       unobserved entries (not used, included for consistency).
    %     params: Struct containing problem parameters.
    %       maxIter: [default: 200].
    %       convThr: [default: 1e-3].
    %       prtLevel: 1=basic per-iteration output [default: 0].
    %       logLevel: 0=basic summary info, 1=detailed per-iteration info
    %         [default: 0]
    %
    %   Returns:
    %     Y: D x N completed data.
    %     history: Struct containing minimal diagnostic info.

    % This needed to make sure mu not passed accidentally, overriding default in alm_mc.
    newparams = struct;
    fields = {'maxIter', 'convThr', 'prtLevel', 'logLevel'};
    defaults = {200, 1e-3, 1, 1};
    for i=1:length(fields)
      if ~isfield(params, fields{i})
        newparams.(fields{i}) = defaults{i};
      else
        newparams.(fields{i}) = params.(fields{i});
      end
    end
    params = newparams;

    groups = self.cluster(C);
    Y = self.X;
    [history.rtime, history.status, history.iter] = deal(0);
    for ii=1:self.n
      indices = find(groups == ii);
      [Y(:, indices), subprob_hist] = alm_mc(self.X(:, indices), ...
          self.Omega(:, indices), params);
      history.rtime = history.rtime + subprob_hist.rtime;
      history.status = max(history.status, subprob_hist.status);
      history.iter = history.iter + subprob_hist.iter;
      if params.logLevel > 0
        history.feas{ii} = subprob_hist.feas;
      end
    end
    end

  end

end
