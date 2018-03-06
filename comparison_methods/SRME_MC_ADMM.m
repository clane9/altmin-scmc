function [X,Z,E]=SRME_MC_ADMM(X0,M,Xinit,lambda,alpha,params)
% X0---matrix with missing entries
% M---indices binary (0-1) matrix, 1/0 indicates observed/missing entries
% Xinit---Initialization for completion X. If empty, will take Xinit=X0.
% lambda and alpha are regularization parameters
% params---struct containing maxIter [default:800], maxTime [default:Inf],
%   convThr [default:1e-6].

% X---recovered matrix
% Z---self-representation coefficients matrix
% E---representation errors matrix

% More details about the algorithm can be found in the following paper:
% Sparse subspace clustering for data with missing entries and high-rank matrix completion
% J Fan, TWS Chow. Neural Networks 2017(93):36-44

% Written by Jicong Fan.
% Edited by Connor Lane:
%   - Added some parameters to control runtime.
%   - Added initialization for X (completion).

if nargin < 6; params = struct; end
fields = {'maxIter', 'maxTime', 'convThr'};
defaults = {800, Inf, 1e-6};
for ii=1:length(fields)
    if ~isfield(params, fields{ii})
        params.(fields{ii}) = defaults{ii};
    end
end
tstart = tic;

[d,n]=size(X0);
%
if isempty(Xinit)
    Xinit=X0;
end
X=Xinit;
Y=X0;
E=zeros(d,n);
Z=zeros(n,n);
A=Z;
Q1=zeros(d,n);
Q2=zeros(n,n);
Q3=zeros(d,n);
%
u=0.1;
u_max=1e10;
r0=1.01;
%
normF_X0=norm(X0,'fro');
%
iter=0;
%
I=eye(n);
for i=1:n
    h(i)=mean(X0(M(:,i)==1,i));
end
g=ones(1,d)/d;
%
while iter<params.maxIter
    iter=iter+1;
    % A_new
    temp=X'*(X-E+Q1/u)+Z-Q2/u;
    A_new=inv(X'*X+I)*temp;
    % Z_new
    temp=A_new+Q2/u;
    J=max(0,temp-1/u)+min(0,temp+1/u);
    Z_new=J-diag(diag(J));
    % Y_new
    Y_new=solve_NuclearNorm(X-Q3/u,u/alpha);
    % X_new
    X_new=((E-Q1/u)*(I-A_new')+(Y_new+Q3/u))*inv((I-A_new)*(I-A_new')+I);
    X_new=X_new.*~M+X0.*M;
    % E_new
    XNZN=X_new*A_new;
    temp=X_new-XNZN+Q1/u;
%     E_new=max(0,temp-1/(u/lambda))+min(0,temp+1/(u/lambda));
    E_new=u/(2*lambda+u)*temp;
    %
    Q1=Q1+u*(X_new-XNZN-E_new);
    Q2=Q2+u*(A_new-Z_new);
    Q3=Q3+u*(Y_new-X_new);
    %
    stopC1=norm(X_new-XNZN-E_new,'fro')/normF_X0;
    stopC2=max([norm(Y_new-X_new,'fro') norm(A_new-Z_new,'fro')])/normF_X0;
    stopC3=max([norm(A_new-A,'fro') norm(Z_new-Z,'fro') norm(X_new-X,'fro') norm(E_new-E,'fro')])/normF_X0;
    %
    isstopC=max([stopC1 stopC2 stopC3])<params.convThr;
    if mod(iter,100)==0||isstopC
        disp(['rankX=' num2str(rank(X_new,1e-3*norm(X_new,2)))])% can be closed for acceleration
        disp(['iteration=' num2str(iter) '/' num2str(params.maxIter)])
        disp(['stopC1=' num2str(stopC1)])
        disp(['stopC2=' num2str(stopC2)])
        disp(['stopC3=' num2str(stopC3)])
        disp(['mu=' num2str(u)])
        disp('......')
    end
    if isstopC
        disp('converged')
        break;
    end
    if toc(tstart) >= params.maxTime
        disp('timeout!')
        break;
    end
    %
    r=r0;
    u=min(u_max,r*u);
    Z=Z_new;
    A=A_new;
    X=X_new;
    Y=Y_new;
    E=E_new;
        %
end

end


%% rank minimization
function [Z]=solve_NuclearNorm(L,mu)
    [U,sigma,V] = svd(L,'econ');
    sigma = diag(sigma);
    svp = length(find(sigma>1/mu));
    if svp>=1
        sigma = sigma(1:svp)-1/mu;
    else
        svp = 1;
        sigma = 0;
    end
    Z=U(:,1:svp)*diag(sigma)*V(:,1:svp)';
end
