%% This is a handwritten character recognition system done with
%% neural network, this implements a one-vs-all classification.
%% This outputs a vector containing all zeros, except for one index
%% which is the prediction of digit recongnition.

% Clean up space to initialize environment
clear;
close all;
clc;

% Setup initial parameters
input_layer_size = 400; % 20x20 input images of digits
hidden_layer_size = 25; % 25 hidden units
num_labels = 10;    % 10 labels from 1 to 10 (NOTE: 0 is labeled as 10)


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


%% Step 2: Loading parameters
fprintf('Loading saved neural network parameters....\n\n');
pause(1);

% Load parameter weights from dataset
load('WeightsData.mat');

fprintf('Weights loaded. Press enter to continue....\n\n');
pause;


%% Step 3: Implement Predicting System
fprintf('Computing predictions....\n\n');
pause(1);

prediction = Predict(X, Theta1, Theta2);

fprintf('Training Set Accuracy: %f\n\n', mean(double(prediction == y)) * 100);

fprintf('Program paused. Press enter to continue....\n\n');
pause;


% Random selection of examples
selection = randperm(m);

for i = 1:m,
    % Display data
    fprintf('Displaying Example Image....\n\n');
    DisplayData(X(selection(i), :));
    
    prediction = Predict(X(selection(i), :), Theta1, Theta2);
    fprintf('Neural Network prediction: %d (digit %d)\n', prediction, mod(prediction, 10));
    
    s = input('Paused - Press enter to continue, q to exit: ', 's');
    
    if s == 'q',
        break;
    endif
endfor
