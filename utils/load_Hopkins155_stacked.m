function [X, groupsTrue] = load_Hopkins155_stacked(datapath, subset)
% load_Hopkins155_all_checker   Load all Hopkins 155 3-motion checker sequences
%   as a single concatenated dataset.
%
%   [X, groupsTrue] = load_Hopkins155_all_checker(datapath, D)
%
%   Args:
%     datapath: Path to Hopkins155 folder containing trajectory subdirectories.
%     subset: Subset of trajectories to include ('checker', 'traffic',
%       'articulated')
%
%   Returns:
%     X: D x (# trajectories) data matrix, where D is the minimum # frames
%       over all trajectories.
%     groups: 1 x (# trajectories) group assignment.

if nargin < 2
  subset = 'checker';
end

if strcmpi(subset, 'checker')
  dir_paths = [dir([datapath '/1*']); ...
      dir([datapath '/2*']); ...
      dir([datapath '/three-cars*'])];
  expected_num_paths = 104;
elseif strcmpi(subset, 'traffic')
  dir_paths = [dir([datapath '/cars*']); ...
      dir([datapath '/truck*']); ...
      dir([datapath '/kanatani1*']); ...
      dir([datapath '/kanatani2*'])];
  expected_num_paths = 38;
elseif strcmpi(subset, 'articulated')
  dir_paths = [dir([datapath '/arm*']); ...
      dir([datapath '/articulated*']); ...
      dir([datapath '/dancing*']); ...
      dir([datapath '/head*']); ...
      dir([datapath '/people*']); ...
      dir([datapath '/two_cranes*']); ...
      dir([datapath '/kanatani3*'])];
  expected_num_paths = 14;
else
  error('Subset %s not supported', subset)
end

% Raise error if all (and only) expected files not found.
num_paths = length(dir_paths);
if num_paths ~= expected_num_paths
  error('Found %d files in %s, expecting %d!', num_paths, datapath, expected_num_paths)
end

[Xs, groupss] = deal(cell(1,num_paths));
[ns, Ds] = deal(zeros(1,num_paths));
traj_idx = 0;
for ii=1:num_paths
  truth_paths = dir([datapath '/' dir_paths(ii).name '/*_truth.mat']);
  if length(truth_paths) >= 1
    % Load data. Variables in *_truth.mat:
    %
    % . width and height: dimensions (in pixels) of all the frames in the
    %   video sequence.
    % . points: number of tracked points P.
    % . frames: number of frames F.
    % . y: a matrix 3xPxF containing the homogeneous coordinates of the P
    %   points in the F frames.
    % . x: a matrix 3xPxF derived from y by normalizing the first two
    %   components of each vector such that they belong to the interval
    %   [-1;1].
    % . K: the 3x3 normalization matrix used to pass from y to x
    %   (x=K^(-1)*x).
    % . s: a Px1 vector containing the ground-truth segmentation; for each
    %   point it gives the index of the corresponding motion group.
    truth_path = [datapath '/' dir_paths(ii).name '/' truth_paths(1).name];
    load(truth_path, 's', 'x');

    % Skip if doesn't have 3 motions.
    % NOTE: this is important since there are many 2-motion sequences
    % derived from the 3-motions.
    n = max(s);
    if n ~= 3
      continue
    end
    % We found one new trajectory to add.
    traj_idx = traj_idx + 1;

    groupss{traj_idx} = s';
    ns(traj_idx) = n;

    N = size(x,2); D = 2*size(x,3);
    Ds(traj_idx) = D;

    % x is 3 x (# points) x (# frames). Use only first two coords and reshape to
    % be (2 * # frames) x (# points).
    % Taken from SSC_ADMM_v1.1 run_SSC_MC
    Xs{traj_idx} = reshape(permute(x(1:2,:,:),[1 3 2]),D,N);
  end
end

% Trim extra empty cells.
num_traj = traj_idx;
Xs = Xs(1:num_traj); groupss = groupss(1:num_traj);
ns = ns(1:num_traj); Ds = Ds(1:num_traj);

% Loop over trajectories, to prepare for stacking.
% Trimming # frames and shifting group labels.
minD = min(Ds);
n = 0; % Total groups.
for ii=1:num_traj
  % Sort for visualization convenience.
  [~, Ind] = sort(groupss{ii});
  Xs{ii} = Xs{ii}(1:minD, Ind);
  groupss{ii} = groupss{ii}(Ind) + n;
  n = n+ns(ii);
end
X = [Xs{:}];
groupsTrue = [groupss{:}]';
end
