%% This is a simple machine learning problem where we design a model to predict
%% the price of a house based on size and number of bedrooms.

% This problem has features x = [x1, x2] and one output y.
% We solve this using linear regression with multiple variable
% a.k.a. multivariate linear regression.

% First we clear our system to run to new regression without garbage
clear;
close all;
clc;

fprintf('Solving with normal equations....\n');

fprintf('Loading data....\n\n');
pause(1);

%% Step 1: Load data from dataset
data = csvread('HousingPriceData.txt');
X = data(:, 1:2);
y = data(:, 3);
m = length(y);

% Add intercept term to X as an extra column
X = [ones(m, 1) X];

fprintf('Program paused. Press enter to continue....\n');
pause;

%% Step 2: Calculate the parameters from the normal equation
fprintf('Runnning normal equations....\n\n');
pause(1);

theta = NormalEquation(X, y);

% Display normal equation's result
fprintf('Theta computed from the normal equations: \n');
fprintf(' %f \n', theta);
fprintf('\n');

fprintf('Program paused. Press enter to continue....\n');
pause;


%% Step 3: Testing for Gradient descent method
fprintf('Running test over a house of 1650-sq-ft and 3 bedrooms....\n\n');
pause(1);

price = [1, 1650, 3] * theta;

fprintf(['Predicted price of a 1650 sq-ft, 3 br house ' ...
         '(using gradient descent):\n $%f\n'], price);
