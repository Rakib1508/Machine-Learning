%% This is a simple machine learning problem where we design a model to predict
%% the price of a house based on size and number of bedrooms.

% This problem has features x = [x1, x2] and one output y.
% We solve this using linear regression with multiple variable
% a.k.a. multivariate linear regression.

% First we clear our system to run to new regression without garbage
clear ; close all; clc

fprintf('Solving with gradient descent....\n');

fprintf('Loading data....\n\n');
pause(1);

%% Step 1: Load data from dataset
data = load('HousingPriceData.txt');
X = data(:, 1:2);
y = data(:, 3);
m = length(y);

fprintf('Program paused. Press enter to continue....\n');
pause;


%% Step 2: Normalize features to a comparable scale
% Scale features and set them to zero mean and std deviation 1
fprintf('Normalizing Features....\n\n');
pause(1);

[X mu sigma] = FeatureNormalizer(X);

% Add intercept term to X as an extra column
X = [ones(m, 1) X];

fprintf('Program paused. Press enter to continue....\n');
pause;


%% Step 3: Gradient descent algorithm to find optimized parameters
fprintf('Running gradient descent....\n\n');
pause(1);

% Choose a learning rate and number of iterations
alpha = 0.01;
iter = 400;

% Initialize parameter vector and run Gradient Descent 
theta = zeros(3, 1);
[theta, J_tracker] = GradientDescent(X, y, theta, alpha, iter);

% Display gradient descent's result
fprintf('Theta computed from gradient descent: \n');
fprintf(' %f \n', theta);
fprintf('\n');


%% Step 4: Plot the convergence graph
figure;
plot(1:numel(J_tracker), J_tracker, '-b', 'LineWidth', 2);
xlabel('Number of iterations');
ylabel('Cost value');

fprintf('Program paused. Press enter to continue....\n');
pause;


%% Step 5: Testing for Gradient descent method
fprintf('Running test over a house of 1650-sq-ft and 3 bedrooms....\n\n');
pause(1);

testCase = [1650, 3];
prediction = (testCase - mu) ./ sigma;
price = [1, prediction] * theta;

fprintf(['Predicted price of a 1650 sq-ft, 3 br house ' ...
         '(using gradient descent):\n $%f\n'], price);
