%% Given a dataset of a microchip production plant where we have previous
%% test reports about the quality of microchips classified into two segments,
%% Accepted and Rejected based on two independent featues (assumed).
%% It's time to train the system with these data and implement logistic
%% regression in order to obtain predictions for future productions.

% Clean up the space to begin regression
clear;
close all;
clc;

fprintf('Loading data....\n\n');
pause(1);


%% Step 1: Load data from dataset
% Load data and divide the features and labels
data = load('MicrochipTestData.txt');
X = data(:, [1, 2]);
y = data(:, 3);

fprintf('Data imported. Press enter to continue....\n\n');
pause;


%% Step 2: Plot data on a graph
fprintf('Plotting data....\n\n');
pause(1);

DataPlotter(X, y);
hold on;

% Labels and legends
xlabel('Microchip test 1');
ylabel('Microchip test 2');

legend('y = 1', 'y = 0');
hold off;

fprintf('Plotting complete. Press enter to continue....\n\n');
pause;


%% Step 3: Perform Regularized Logistic Regression
fprintf('Computing regularized cost and gradient....\n\n');
pause(1);

% Prepare data along with adding intercept terminal_size
X = MapFeatures(X(:, 1), X(:, 2));

% Initialize necessary variables
initial_theta = zeros(size(X, 2), 1);
lambda = 1; % regularization parameter

% Compute initial cost and gradient for the regression
[cost, grad] = CostFunction(X, y, initial_theta, lambda);

% Display initial cost and gradient
fprintf('Cost at initial theta (zeros): %f\n', cost);
fprintf('Expected cost (approx): 0.693\n');
fprintf('Gradient at initial theta (zeros) - first five values only:\n');
fprintf(' %f \n', grad(1:5));
fprintf('Expected gradients (approx) - first five values only:\n');
fprintf(' 0.0085\n 0.0188\n 0.0001\n 0.0503\n 0.0115\n\n');

fprintf('Program paused. Press enter to continue....\n\n');
pause;


%% Step 4: Non-zero initialization of theta to run cost function
fprintf('Running cost function....\n\n');
pause(1);

% set theta to a non-zero value and compute cost and gradient
test_theta = ones(size(X, 2), 1);
[cost, grad] = CostFunction(X, y, test_theta, lambda);

% Display cost and gradient for non-zero initialization of theta.
fprintf('\nCost at test theta (with lambda = 10): %f\n', cost);
fprintf('Expected cost (approx): 3.16\n');
fprintf('Gradient at test theta - first five values only:\n');
fprintf(' %f \n', grad(1:5));
fprintf('Expected gradients (approx) - first five values only:\n');
fprintf(' 0.3460\n 0.1614\n 0.1948\n 0.2269\n 0.0922\n\n');

fprintf('Program paused. Press enter to continue.\n');
pause;


%% Step 5: Optimize theta parameters with advanced optimization algorithm
fprintf('Running advanced optimization algorithm....\n\n');
pause(1);

% Re-initialize parameters
initial_theta = zeros(size(X, 2), 1);
lambda = 1;

options = optimset('GradObj', 'on', 'MaxIter', 400);

% Run optimization
[theta, J, exit_flag] = fminunc(@(t)(CostFunction(X, y, t, lambda)), initial_theta, options);

fprintf('Parameters optimized. Press enter to continue....\n\n');
pause;


%% Step 6: Plot decision boundary on visual graphical representation
fprintf('Plotting decision boundary....\n\n');
pause(1);

% Plot data and render title
PlotDecisionBoundary(X, y, theta);
hold on;
title(sprintf('lambda = %g', lambda))

% Labels and Legend
xlabel('Microchip Test 1')
ylabel('Microchip Test 2')

legend('y = 1', 'y = 0', 'Decision boundary')
hold off;

fprintf('Data plotted with decision boundary. Press enter to continue....\n\n');
pause;


%% Step 7: Compute training accuracy
fprintf('Computing training accuracy....\n\n');
pause(1);

accuracy = Predict(theta, X);

fprintf('Training accuracy: %f\n', mean(double(accuracy == y)) * 100);
fprintf('Expected Accuracy (with lambda = 1): 83.1 (approx)\n');
