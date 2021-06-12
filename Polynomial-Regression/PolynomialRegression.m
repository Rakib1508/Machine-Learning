%% With a given dataset about the rate of water flowing out of a dam
%% in contrast with the level of water in the reservoir, we need to create
%% a machine learning system to learn this data efficiently with
%% polynomial regression methodology.

% Clean up the space to start the system
clear;
close all;
clc;

%% Step 1: Load and visualize data
fprintf('Loading and visualizing dataset....\n\n');
pause(1);

% Load data from dataset
load('WaterFlowData.mat');

% Plot training data into a graph
plot(X, y, 'rx', 'MarkerSize', 10, 'LineWidth', 1.5);
xlabel('Change in water level (x)');
ylabel('Water flowing out of dam (y)');

fprintf('Data loaded. Press enter to continue....\n\n');
pause;


%% Step 2: Regularize Linear regression cost & gradient
fprintf('Running linear regression....\n\n');
pause(1);

% Initialize regression environment
m = size(X, 1);
theta = [1; 1];

[J, grad] = CostFunction([ones(m, 1) X], y, theta, 1);

fprintf(['Cost at theta = [1 ; 1]: %f\n'...
         '(this value should be about 303.993192)\n\n'], J);

fprintf(['Gradient at theta = [1 ; 1]:  [%f; %f]\n'...
         '(this value should be about [-15.303016; 598.250744])\n\n'], ...
         grad(1), grad(2));

fprintf('Program paused. Press enter to continue....\n\n');
pause;


%% Step 3: Train linear regression with training dataset
fprintf('Training and visualizing training data....\n\n');
pause(1);

% train data with no regularization
lambda = 0;
[theta] = Trainer([ones(m, 1) X], y, lambda);

% Plot current best fit line
plot(X, y, 'rx', 'MarkerSize', 10, 'LineWidth', 1.5);
xlabel('Change in water level (x)');
ylabel('Water flowing out of the dam (y)');
hold on;

plot(X, [ones(m, 1) X] * theta, '--', 'LineWidth', 2);
hold off;

fprintf('Program paused. Press enter to continue....\n\n');
pause;


%% Step 4: Produce Learning curve
fprintf('Producing learning curve....\n\n');
pause(1);

lambda = 0;
[error_train, error_cv] = LearningCurve([ones(m, 1) X], y, ...
            [ones(size(Xval, 1), 1) Xval], yval, lambda);

% Plot to view learning curve
plot(1:m, error_train, 1:m, error_cv);
title('Learning curve diagram');
legend('Training set', 'Cross-validation set');
xlabel('Number of training examples');
ylabel('Error');
axis([0 30 0 150]);

fprintf('\n#Training examples\tTrain error\tCross validation error\n\n');
for i = 1:m,
    fprintf('  \t%d\t\t%f\t%f\n\n', i, error_train(i), error_cv(i));
endfor

fprintf('Program paused. Press enter to continue....\n\n');
pause;


%% Step 5: Feature mapping for polynomial regression
fprintf('Normalizing features....\n\n');
pause(1);

p = 8;

% Map X onto Polynomial Features and Normalize
X_p = PolynomialFeatures(X, p);
[X_p, mu, sigma] = FeatureNormalization(X_p);   % Normalize
X_p = [ones(m, 1), X_p];                        % Add Ones

% Map X_poly_test and normalize (using mu and sigma)
X_poly_test = PolynomialFeatures(Xtest, p);
X_poly_test = bsxfun(@minus, X_poly_test, mu);
X_poly_test = bsxfun(@rdivide, X_poly_test, sigma);
X_poly_test = [ones(size(X_poly_test, 1), 1), X_poly_test];         % Add Ones

% Map X_poly_val and normalize (using mu and sigma)
X_poly_val = PolynomialFeatures(Xval, p);
X_poly_val = bsxfun(@minus, X_poly_val, mu);
X_poly_val = bsxfun(@rdivide, X_poly_val, sigma);
X_poly_val = [ones(size(X_poly_val, 1), 1), X_poly_val];           % Add Ones

fprintf('Normalized Training Example 1:\n\n');
fprintf('  %f  \n', X_p(1, :));

fprintf('\nProgram paused. Press enter to continue....\n\n');
pause;


%% Step 6: Learning curve for polynomial regression
fprintf('Plotting learning curve....\n\n');
pause(1);

lambda = 0;
[theta] = Trainer(X_p, y, lambda);

% Plot training data along with best fit
figure(1);
plot(X, y, 'rx', 'MarkerSize', 10, 'LineWidth', 1.5);
PlotFit(min(X), max(X), mu, sigma, theta, p);
xlabel('Change in water level (x)');
ylabel('Water flowing out of the dam (y)');\
title(sprintf('Polynomial regression fit (lambda = %f)', lambda));

figure(2);
[error_train, error_cv] = LearningCurve([ones(m, 1) X], y, ...
            [ones(size(Xval, 1), 1) Xval], yval, lambda);

plot(1:m, error_train, 1:m, error_cv);
title(sprintf('Polynomial regression learning curve (lambda = %f)', lambda));
xlabel('Number of training examples');
ylabel('Error');
axis([0 13 0 100]);
legend('Training set', 'Cross-validation set');

fprintf('\nPolynomial Regression (lambda = %f)\n\n', lambda);
fprintf('# Training Examples\tTrain Error\tCross Validation Error\n');
for i = 1:m
    fprintf('  \t%d\t\t%f\t%f\n', i, error_train(i), error_cv(i));
end

fprintf('\nProgram paused. Press enter to continue....\n\n');
pause;


%% Step 7: Validation for choosing regularization parameter
fprintf('Optimizing lambda....\n\n');
pause(1);

[lambda_vec, error_train, error_cv] = ValidationCurve(X_p, y, X_poly_val, yval);

close all;
plot(lambda_vec, error_train, lambda_vec, error_cv);
legend('Training set', 'Cross-validation set');
xlabel('lambda');
ylabel('error');

fprintf('\nlambda\t\tTrain Error\tValidation Error\n\n');
for i = 1:length(lambda_vec)
	fprintf(' %f\t%f\t%f\n', ...
            lambda_vec(i), error_train(i), error_cv(i));
end
