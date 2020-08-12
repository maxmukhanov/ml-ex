function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

hipothesis=sigmoid(X*theta);
% You need to return the following variables correctly 
regularization_cost = lambda*sum(theta(2:length(theta)).^2)/(2*m);

J = (sum(-y.*log(hipothesis).-(1-y).*(log(1-hipothesis)))/m) + regularization_cost;
grad = (1/m)*sum(repmat((hipothesis-y), 1, size(X, 2)).*X);

theta_size = length(theta);
gr=grad(:,2:theta_size);


gr=gr.+(theta(2:theta_size,:)'.*lambda./m);

grad = [grad(:,1) gr];
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta






% =============================================================

end
