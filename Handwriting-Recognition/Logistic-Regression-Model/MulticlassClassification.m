%% This is a handwritten character recognition system done with
%% multiclass classification, a.k.a. one-vs-all logistic regression.
%% What this does is basically take one class as 1 and
%% consider all others as 0 from perspective of that specific class.

% Clean up space to initialize environment
clear;
close all;
clc;

% Setup initial parameters
input_layer_size = 400;
num_labels = 10;


%% Step 1: Load and visualize data
fprintf('Loading and visualizing data....\n\n');
pause(1);

% Load training data from dataset
load('HandwrittenDigits.mat');
m = size(X, 1);

% Randomly choose 100 example sets from dataset
random_indices = randperm(m);
example_set = X(random_indices(1:100), :);

DisplayData(example_set);

fprintf('Load complete. Press enter to continue....\n\n');
pause;


%% Step 2: Vectorize logistic regression
fprintf('Testing Cost function with regularization....\n\n');
pause(1);

% Test case for cost function
theta_trial = [-2; -1; 1; 2];
X_trial = [ones(5, 1) reshape(1:15, 5, 3) / 10];
y_trial = ([1; 0; 1; 0; 1] >= 0.5);
lambda_trial = 3;

[J grad] = CostFunction(X_trial, y_trial, theta_trial, lambda_trial);

% Display necessary information
fprintf('Cost: %f\n', J);
fprintf('Expected cost: 2.534819\n');
fprintf('Gradients:\n');
fprintf(' %f \n', grad);
fprintf('Expected gradients:\n');
fprintf(' 0.146561\n -0.548558\n 0.724722\n 1.398003\n\n');

fprintf('Program paused. Press enter to continue....\n\n');
pause;


%% Step 3: One-vs-all training
fprintf('Training One-vs-All Logistic Regression....\n\n');
pause(1);

lambda = 0.1;
[final_theta] = OneVsAll(X, y, num_labels, lambda);

fprintf('\n');
fprintf('Program paused. Press enter to continue....\n\n');
pause;


%% Step 4: Predict for one-vs-all
fprintf('Calculating predictions....\n\n');
pause(1);

prediction = PredictOneVsAll(X, final_theta);

fprintf('Training set accuracy: %f\n', mean(double(prediction == y)) * 100);
