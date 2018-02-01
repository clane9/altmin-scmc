classdef ENSC_Group_MC < ENSC_MC
% ENSC_Group_MC   Solver for alternating elastic-net subspace clustering (with
%   projection), and group-wise completion.
%
%   solver = ENSC_Group_MC(X, Omega, n, lambda, gamma)

  properties
  
  end


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
    
    function [Y, history] = compY(self, ~, C, ~, ~)
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
    %
    %   Returns:
    %     Y: D x N completed data.
    %     history: Struct containing minimal diagnostic info.
    tstart = tic;
    groups = self.cluster(C);
    Y = self.X;
    for ii=1:self.n
      indices = find(groups == ii);
      % TODO: Should modify alm_mc so that it returns some diagnostic
      % info, is more consistent with other methods.
      Y(:, indices) = alm_mc(self.X(:, indices), self.Omega(:, indices));
    end
    history.status = 0; history.iter = 0; history.rtime = toc(tstart);
    end

  end

end
