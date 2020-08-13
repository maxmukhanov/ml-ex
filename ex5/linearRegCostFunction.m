function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

hipothesis = X*theta;


%size(X)
%size(theta)
%size(y)

% You need to return the following variables correctly 

theta_tmp = theta;
theta_tmp(1)=0;

J = sum((hipothesis.-y).^2)/(2*m) + lambda*(sum(theta_tmp.^2))/(2*m);
grad = sum(repmat((hipothesis.-y), 1, size(X,2)).*X)/m;

%size(grad)
reg = lambda*theta_tmp'/m;
%size(reg)
grad = grad + reg;


% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%












% =========================================================================

grad = grad(:);

end
