function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

C_options = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
sigma_options = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
%C_options = [0.01; 0.03];
%sigma_options = [0.01; 0.03];
C_length = length(C_options);
sigma_length = length(sigma_options);


err_min = inf;

for i=1:C_length,
  for j=1:sigma_length,
    model= svmTrain(X, y, C_options(i), @(x1, x2) gaussianKernel(x1, x2, sigma_options(j)));
    predictions = svmPredict(model, Xval);
    err = mean(double(predictions ~= yval));
    if (err < err_min) 
      C = C_options(i);
      sigma = sigma_options(j);
      err_min = err;
    end;
  end;
end;




% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%







% =========================================================================

end
