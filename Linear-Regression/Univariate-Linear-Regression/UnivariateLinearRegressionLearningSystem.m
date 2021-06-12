%% This is a simple machine learning problem where we design a model to predict
%% the profit made from food truck business over each 10000 people in a city.

% This problem has only one feature, x and one output, y.
% So, we solve this using linear regression with one variable
% a.k.a. univariate linear regression.

% First we clear our system to run to new regression without garbage
clear;
close all;
clc;

fprintf('Plotting data....\n');
pause(2);
%% Step 1: Load the dataset. For now, we only have one dataset.
data = load('FoodTruckProfitData.txt');



%% Step 2: Let's plot our dataset into a graph
%          to get an idea of what we are dealing with.

%% Initialization of feature vectors, output & other important variables
X = data(:, 1); % first column is the only feature vector x
y = data(:, 2); % second column is the output vector y
m = length(y);  % the number of training examples for our learning model

%% Plot the data into a graph taking values from feature vector and output
DataPlotter(X, y);

% Pause to view the data in the graph
fprintf('Program paused. Press enter to continue....\n\n');
pause;



%% Step 3: Compute cost function and minimize parameters using gradient descent
X = [ones(m, 1), X];    % add a column of 1 as X_0 vector
theta = zeros(size(data, 2), 1);    % take a vector of zeros as parameters
% size of parameters: [columns in X * 1]

% Gradient descent settings initialization
iter = 1500;    % maximum iterations to calculate minimum cost value
alpha = 0.01;   % learning rate

fprintf('Testing cost function....\n');
pause(2);

% Compute cost according to the initial parameters and display
J = CalculateCostValue(X, y, theta);
fprintf('With theta = [0; 0]\nCost value = %f\n', J);
fprintf('Expected cost value is 32.07 (approx)\n\n');

% Let's initialize theta to a non-zero value
theta = [-1; 2];
J = CalculateCostValue(X, y, theta);

fprintf('With theta = [-1; 2]\nCost value = %f\n', J);
fprintf('Expected cost value is 54.24 (approx)\n\n');

fprintf('Program paused. Press enter to continue....\n\n');
pause;

% Run gradient descent to calculate the parameter for minimum cost value
fprintf('Running gradient descent algorithm....\n\n');
pause(2);

% Gradient descent will return optimized value of parameters
theta = GradientDescent(X, y, theta, alpha, iter);

fprintf('Theta after gradient descent: \n');
fprintf('%f\n', theta);
fprintf('Expected theta values (approx)\n');
fprintf('-3.7097\n1.1743\n\n');

%% Step 4: Plotting best linear fit obtained after gradient descent
hold on;    % keep previous graph visible
plot(X(:, 2), X * theta, '-'); % plot x against predicted values x * theta
legend('Training data', 'Linear regression');
hold off;   % don't overlay further plots on this graph

fprintf('Program paused. Press enter to continue....\n');
pause;

% Run some test cases
fprintf('Running test cases....\n\n');
pause(2);

testCase1 = [1, 3.5];
prediction1 = testCase1 * theta;
fprintf('For population = 35,000, profit predicted = %f\n',...
         prediction1 * 10000);

testCase2 = [1, 7];
prediction2 = testCase2 * theta;
fprintf('For population = 70,000, profit predicted = %f\n\n',...
         prediction2 * 10000);

fprintf('Program paused. Press enter to continue....\n');
pause;


%% Step 5: Visualize stepwise decrease of cost function
fprintf('Visualizing J(theta_0, theta_1)....\n\n');

% grid to show the calculated value of J
theta0 = linspace(-10, 10, 100);
theta1 = linspace(-1, 4, 100);

% initialize a matrix of zeros to be updated and plotted later
J_values = zeros(length(theta0), length(theta1));

% get values of J_values
for i = 1:length(theta0),
    for j = 1:length(theta1),
        
        theta_ij = [theta0(i); theta1(j)];
        J_values(i, j) = CalculateCostValue(X, y, theta_ij);
        
    endfor
endfor

% Transpose J_values in order to use it in surf command
J_values = J_values';

% Plot 3D surface graph
figure;
surf(theta0, theta1, J_values);
xlabel('\theta0');
ylabel('\theta1');

% Plot 2D contour graph
figure;

% Plot J_values as 15 contours spaced logarithmically between 0.01 and 100
contour(theta0, theta1, J_values, logspace(-2, 3, 20));
xlabel('\theta0');
ylabel('\theta1');
hold on;

plot(theta(1), theta(2), 'rx', 'MarkerSize', 10, 'LineWidth', 2);
