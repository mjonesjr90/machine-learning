function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

h = sigmoid(X*theta);
theta_after_one = theta(2:size(theta)); %make new matrix with theta after 1
new_theta = [0; theta_after_one]; %set first row to 0 and the remaining to new theta

% Dont forget to use elementwise arithmetic
reg = (lambda/(2*m))*new_theta'*new_theta;
J = (1/m)*sum((-(y).*log(h)) - ((1-y).*log(1-h))) + reg;

grad = (1/m)*(X'*(h-y)+lambda*new_theta);




% =============================================================

end
