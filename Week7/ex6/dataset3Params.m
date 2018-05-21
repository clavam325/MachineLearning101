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

C_row    =[0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
sigma_row=[0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];

for i=1:length(C_row)
    C = C_row(i);
    for j=1:length(sigma_row)
        sigma=sigma_row(j);
        model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma)); 
        predictions = svmPredict(model,Xval);

        error(j) = mean(double(predictions ~= yval));

% Accumulating error from i=1:m
        if (j==1)
            error_j=error(j);
        else
            error_j=[error_j; error(j)];
        end
    end
    if (i==1)
        error_i=error_j;
    else
        error_i=[error_i error_j];
    end            
    
end

[r,c]=find(error_i==min(min(error_i)));


C=C_row (c);
sigma=sigma_row(r);






% =========================================================================

end
