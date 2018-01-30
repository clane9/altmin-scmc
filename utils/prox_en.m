function a = prox_en(x, gamma, lamb)
% prox_en    Evaluate proximal operator of elastic net.
%   \gamma*||x||_1 + (1-gamma)/2||x||_2^2
%
%   a = prox_en(x, gamma, lamb)
lamb1 = gamma*lamb; lamb2 = (1-gamma)*lamb;
xprime = x / (1 + lamb2);
if gamma > 0
  a = sign(xprime).*max(abs(xprime) - lamb1/(1+lamb2), 0);
else
  a = xprime;
end
end
