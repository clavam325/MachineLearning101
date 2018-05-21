function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

%fprintf('Size of X %f X %f.\n', size(X,1), size(X,2));
%fprintf('Size of Y %f X %f.\n', size(y,1), size(y,2));
%fprintf('Size of theta %f X %f.\n', size(theta,1), size(theta,2));


h = X*theta;
thetatemp = theta(2:end, 1);
J = (1/(2*m)) .* ( sum((h - y).^2) + (lambda) .* sum((thetatemp).^2) );

thetatemp = theta;
thetatemp(1) = 0;
grad = (1/m) .* (X' * (h-y) + lambda .* thetatemp);

% =========================================================================

grad = grad(:);

end
