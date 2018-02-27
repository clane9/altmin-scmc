function [groups, Z, X, history]=SRME_MC_ADMM_ron(X0,Omega,n,lambda,varargin)
% X0---matrix with missing entries
% Omega---indices binary (0-1) matrix, 1/0 indicates observed/missing entries
% lambda and alpha are regularization parameters

% X---recovered matrix
% Z---self-representation coefficients matrix
% E---representation errors matrix

alpha = 1;
if nargin >= 5; alpha = varargin{1}; end

[D,N]=size(X0);
%
X=X0;
Y=X0;
E=zeros(D,N);
Z=zeros(N,N);
A=Z;
Q1=zeros(D,N);
Q2=zeros(N,N);
Q3=zeros(D,N);
%
maxIter=800;
u=0.1;
u_max=1e10;
r0=1.01;
e=1e-6;
updateHistory = (nargout > 3); % Prevents checking arguments in the loop
%
normF_X0=norm(X0,'fro');
%
iter=0;
%
I=eye(N);
for i=1:N
    h(i)=mean(X0(Omega(:,i)==1,i));
end
g=ones(1,D)/D;
isstopC = 0;
%
while iter<maxIter && ~isstopC
    iter=iter+1;
    % A_new
    temp=(lambda*X'*X)+u*(Z-Q2/u);
    A_new=inv(lambda*X'*X+u*I)*temp;
    % Z_new
    temp=A_new+Q2/u;
    Z_new = max(0,(abs(temp) - 1/u)) .* sign(temp);
    Z_new = Z_new - diag(diag(Z_new));

    % Y_new
    Y_new=solve_NuclearNorm(X-Q3/u,u/alpha);
    % X_new
    X_new = u*((Y_new+Q3/u)) / (lambda*(I-A_new)*(I-A_new')+u*I);
    X_new = X_new.*~Omega+X0.*Omega;

    Q2=Q2+u*(A_new-Z_new);
    Q3=Q3+u*(Y_new-X_new);
    %
    stopC2=max([norm(Y_new-X_new,'fro') norm(A_new-Z_new,'fro')])/normF_X0;
    stopC3=max([norm(A_new-A,'fro') norm(Z_new-Z,'fro') norm(X_new-X,'fro')])/normF_X0;
    %
    isstopC=max([stopC2 stopC3])<e;

    r=r0;
    u=min(u_max,r*u);
    Z=Z_new;
    A=A_new;
    X=X_new;
    Y=Y_new;

    if updateHistory 
        % history.stop_cond(k) = stop_cond;
        history.medCspr(iter) = median(1.0 - mean(abs(Z) > 1e-3));
        history.reconerr(iter) = mean(sum((X - X*Z).^2));
    end
end

% Spectral clustering.
Csym = build_affinity(Z - diag(diag(Z))); % CL edit here.
groups = spectral_clustering(Csym, n);
      
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


