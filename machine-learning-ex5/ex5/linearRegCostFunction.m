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
h = X * theta; % The hypothesis
error = h - y; % The difference between the hypothesis and the real y
error_sq = error.^2;
J = 1/(2*m) * sum(error_sq); % Unregularized J
grad = (1/m) * (X' * error); % Unregularized grad; Ignore alpha in this example

theta_after_one = theta(2:size(theta)); %make new matrix with theta after 1
new_theta = [0; theta_after_one]; %set first row to 0 and the remaining to new theta
reg_J = (lambda/(2*m))*sum(new_theta.^2);
reg_Grad = (lambda/m)*new_theta;
J = J + reg_J;
grad = grad + reg_Grad;
% =========================================================================

grad = grad(:);

end
