classdef ENSC_Group_MC < ENSC_MC

  properties
  
  end


  methods
    
    function self = ENSC_Group_MC(X, Omega, n, lambda, gamma)
    %   Args:
    %     X: D x N incomplete data matrix.
    %     Omega: D x N binary pattern of missing entries.
    %     n: number of clusters.
    self = self@ENSC_MC(X, Omega, n, lambda, gamma);
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
    groups = self.cluster(C);
    Y = self.X;
    for ii=1:self.n
      indices = find(groups == ii);
      % TODO: Should modify simple_alm_mc so that it returns some diagnostic
      % info, is more consistent with other methods.
      Y(:, indices) = alm_mc(self.X(:, indices), self.Omega(:, indices));
    end
    history.status = 0;
    end

  end

end
