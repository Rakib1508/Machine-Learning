%% This is a handwritten character recognition system done with
%% neural network, this implements a one-vs-all classification.
%% This outputs a vector containing all zeros, except for one index
%% which is the prediction of digit recongnition.

%% This version of neural network is implemented using hidden layer and
%% forward-backward propagation algorithms.

% Clean up space to initialize environment
clear;
close all;
clc;

% Setup initial parameters
input_layer_size = 400;
hidden_layer_size = 25;
num_labels = 10;    % 10 labels, from 1 to 10   
                    % (note that we have mapped "0" to label 10)

%% Step 1: Load & visualize data from training dataset
fprintf('Loading necessary data....\n\n');
pause(1);

% Load training data
load('DigitsData.mat');
m = size(X, 1);

fprintf('Data loaded successfully. Press enter to continue....\n\n');
pause;

% Randomly choose 100 examples from the training set to display
selection = randperm(size(X, 1));
selection = selection(1:100);

fprintf('Visualizing demo examples....\n\n');
pause(1);

DisplayData(X(selection, :));

fprintf('Visualization complete. Press enter to continue....\n\n');
pause;


%% Step 2: Load parameters for hidden layers
fprintf('Loading necessary parameters for hidden layers....\n\n');
pause(1);

% Load parameters Theta1 for 1st hidden layer & Theta 2 for output layer
load('WeightsData.mat');

% Unroll parameters
parameters = [Theta1(:); Theta2(:)];

fprintf('Parameters loaded. Press enter to continue....\n\n');
pause;


%% Step 3: Compute cost value using forward propagation
fprintf('Running forward propagation....\n\n');
pause(1);

% Initialize regularization parameters
lambda = 0;

J = CostFunction(parameters, input_layer_size, hidden_layer_size,...
                 num_labels, X, y, lambda);

fprintf(['Cost at parameters (loaded from ex4weights): %f\n'...
         '(this value should be about 0.287629)\n\n'], J);

fprintf('Program paused. Press enter to continue....\n\n');
pause;


%% Step 4: Implement regularization
% With cost function successfully implemented, we will try to
% implement regularization alongside the cost function.
fprintf('Calculating cost function with regularization....\n\n');
pause(1);

% set regularization parameter
lambda = 1;

J = CostFunction(parameters, input_layer_size, hidden_layer_size, ...
                 num_labels, X, y, lambda);

fprintf(['Cost at parameters (loaded from ex4weights): %f\n'...
         '(this value should be about 0.383770)\n\n'], J);

fprintf('Program paused. Press enter to continue....\n\n');
pause;


%% Step 5: Perform Sigmoid gradient
% To calculate gradients in neural network, we need to calculate
% the gradient of the sigmoid function first.
fprintf('Computing sigmoid gradient....\n\n');
pause(1);

g = SigmoidGradient([-1 -0.5 0.5 1]);
fprintf('Sigmoid gradient evaluated at [-1 -0.5 0 0.5 1]:\n  ');
fprintf('%f ', g);
fprintf('\n\n');

fprintf('Program paused. Press enter to continue....\n\n');
pause;


%% Step 6: Initializing layer-wise parameters
% We will be starting to implment a two layer neural network
% that classifies digits, starting by implementing a function
% to initialize the weights of the neural network
fprintf('Initializing parameters for neural network....\n\n');
pause(1);

initial_theta1 = RandomInitialization(input_layer_size, hidden_layer_size);
initial_theta2 = RandomInitialization(hidden_layer_size, num_labels);

% Unroll parameters into a vector
initial_parameters = [initial_theta1(:); initial_theta2(:)];

fprintf('Initialization complete. Press enter to continue....\n\n');
pause;


%% Step 7: Implement backward propagation
fprintf('Running back propagation....\n\n');
pause(1);

% check gradients to ensure correct implementation of backprop
CheckGradients;

fprintf('Program paused. Press enter to continue....\n\n');
pause;


%% Step 8: Implement regularization
% With proper implementation of backprop, now it is possible to 
% implement regularization over cost and gradients.
fprintf('Checking Back-propagation with regularization....\n\n');
pause(1);

% Check gradients with regularization parameter
lambda = 3;
CheckGradients(lambda);

% Get cost value for debugging
debug_J = CostFunction(parameters, input_layer_size, hidden_layer_size, ...
                       num_labels, X, y, lambda);

fprintf(['Cost at (fixed) debugging parameters (w/ lambda = %f): %f\n' ...
         '(for lambda = 3, this value should be about 0.576051)\n\n'], lambda, debug_J);

fprintf('Program paused. Press enter to continue....\n\n');
pause;


%% Step 9: Train neural network
% Train neural network using optimized parameters obtained by using
% the function fmincg.
fprintf('Training neural network....\n\n');
pause(1);

% Set the settings for fmincg
options = optimset('MaxIter', 50);
lambda = 1;

% create a shorthand for calling fmincg
cf = @(p)CostFunction(p, input_layer_size, hidden_layer_size, num_labels, ...
                      X, y, lambda);

[params, cost] = fmincg(cf, initial_parameters, options);

% Unroll parameters
Theta1 = reshape(params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));
Theta2 = reshape(params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

fprintf('Program paused. Press enter to continue....\n\n');
pause;


%% Step 10: Visulize weights
fprintf('Visualizing Neural network....\n\n');
pause(1);

DisplayData(Theta1(:, 2:end));

fprintf('Program paused. Press enter to continue....\n\n');
pause;


%% Step 11: Testing
% Try predicting some values after training the neural network
fprintf('Testing new data....\n\n');
pause(1);

prediction = Predict(X, Theta1, Theta2);

fprintf('Training Set accuracy: %f\n', mean(double(prediction == y)) * 100);
