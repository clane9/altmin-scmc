function Trajs = load_Hopkins155(datapath)
% load_Hopkins155   Load Hopkins 155 motion segmentation dataset.
%
%   Trajs = load_Hopkins155(datapath)
%
%   Args:
%     datapath: Path to Hopkins155 folder containing trajectory subdirectories.
%
%   Returns:
%     Trajs: 1 x 156 cell array containing 1 struct per trajectory, each with
%       fields: s (segmentation), n (num groups), N (num tracked features), D
%       (num frames), and X (D x N data matrix).

% Glob list of files in data path, among which will hopefully be all the
% trajectory subfolders.
% NOTE: can't get data paths directly since recursive patterns e.g.
% */*_truth.mat don't work.
dir_paths = dir([datapath '/*']);
num_paths = length(dir_paths);

% Raise error if all (and only) expected files not found.
if num_paths < 156
  error('Only found %d files in %s, expecting 156!', num_paths, datapath)
end

Trajs = cell(1,num_paths);
traj_idx = 0;
for ii=1:num_paths
  truth_paths = dir([datapath '/' dir_paths(ii).name '/*_truth.mat']);
  if length(truth_paths) >= 1
    % We found one new trajectory to add.
    traj_idx = traj_idx + 1;

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

    Traj.name = dir_paths(ii).name;
    Traj.s = s;
    Traj.n = max(s);
    N = size(x,2);
    D = 2*size(x,3);

    % x is 3 x (# points) x (# frames). Use only first two coords and reshape to
    % be (2 * # frames) x (# points).
    % Taken from SSC_ADMM_v1.1 run_SSC_MC
    Traj.X = reshape(permute(x(1:2,:,:),[1 3 2]),D,N);
    Trajs{traj_idx} = Traj;
  end
end

% Trim extra empty cells.
Trajs = Trajs(1:traj_idx);
end
