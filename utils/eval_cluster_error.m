function [err, groups] = eval_cluster_error(groups, true_groups)
groups = bestMap(true_groups, groups);
err = 1.0 - mean(true_groups == groups);
end
