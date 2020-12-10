%% Given a set of scores from two exams of many students, it is required to 
%% predict whether the student will be admitted into the university.
%% To solve this problem, we will apply binary classification with
%% logistic regression.

% Initialize environment via cleaning the workspace
clear;
close all;
clc;

fprintf('Loading data....\n');
pause(1);


%% Step 1: Load data from dataset
data = load('StudentExamRecords.txt');
X = data(:, [1:2]);
y = data(:, 3);
m = length(y);

fprintf('Load complete. Press enter to continue....\n\n');
pause;


%% Step 2: Plot the data into the graph to get idea about the dataset
fprintf(['Plotting data with + indicating (y = 1) examples and '...
         'o indicating (y = 0) examples.\n']);
pause(1);

% Plot data into graph
DataPlotter(X, y);

% Keep the graph still
hold on;

% Do some labeling & legends, legends are taken in specified order
xlabel('Exam 1 Score');
ylabel('Exam 2 Score');
legend('Admitted', 'Not admitted');
hold off;

fprintf('Graphing complete. Press enter to continue....\n\n');
pause;


%% Step 3: Compute cost value and gradient
fprintf('Computing cost value and gradient....\n');
pause(1);

%  Setup the data matrix and then add intercept term of ones
[m, n] = size(X)
X = [ones(m, 1) X]; % add a column of ones as first column

% Initialize theta parameter vector
initial_theta = zeros(n+1, 1);

% Compute the cost value and gradient
[cost, grad] = CostFunction(X, y, initial_theta);

% display necessary details
fprintf('Cost at initial theta (zeros): %f\n', cost);
fprintf('Expected cost (approx): 0.693\n');
fprintf('Gradient at initial theta (zeros): \n');
fprintf(' %f \n', grad);
fprintf('Expected gradients (approx):\n -0.1000\n -12.0092\n -11.2628\n\n');

fprintf('Program paused. Press enter to continue....\n\n');
pause;

% Try different value of theta for initialization
fprintf('Trying different initial value of theta....\n');
pause(1);

test_theta = [-24; 0.2; 0.2];
[cost, grad] = CostFunction(X, y, test_theta);

fprintf('Cost at test theta: %f\n', cost);
fprintf('Expected cost (approx): 0.218\n');
fprintf('Gradient at test theta: \n');
fprintf(' %f \n', grad);
fprintf('Expected gradients (approx):\n 0.043\n 2.566\n 2.647\n\n');

fprintf('Program paused. Press enter to continue....\n\n');
pause;


%% Step 4: Optimize theta using fminunc function
fprintf('Optimizing theta....\n\n');
pause(1);

% Set option parameters for optimizing function
options = optimset('GradObj', 'on', 'MaxIter', 400);

% Perform optimization
[theta, cost] = fminunc(@(t)(CostFunction(X, y, t)), initial_theta, options);

% Print results on screen
fprintf('Cost at theta found by fminunc: %f\n', cost);
fprintf('Expected cost (approx): 0.203\n');
fprintf('theta: \n');
fprintf(' %f \n', theta);
fprintf('Expected theta (approx):\n');
fprintf(' -25.161\n 0.206\n 0.201\n\n');

fprintf('Program paused. Press enter to continue....\n\n');
pause;


%% Step 5: Plot decision boundary
fprintf('Plotting dataset and decision boundary....\n');
pause(1);

PlotDecisionBoundary(X, y, theta);
hold on;
xlabel('Exam 1 Score');
ylabel('Exam 2 Score');

legend('Admitted', 'Not admitted');
hold off;

fprintf('Plotting finished. Press enter to continue....\n\n');
pause;


%% Step 6: Testing on unseen data
fprintf('Running test case,\nExam 1 Score = 45,\nExam 2 Score = 85....\n\n');
pause(1);

testCase = [1 45 85];
result = Sigmoid(testCase * theta);
fprintf(['For a student with scores 45 and 85, we predict an admission ' ...
         'probability of %f\n'], result);
fprintf('Expected value: 0.775 +/- 0.002\n\n');

% Compute accuracy of the training set
accuracy = Predict(theta, X);

fprintf('Training accuracy: %f\n', mean(double(accuracy == y)) * 100);
fprintf('Expected accuracy (approx): 89.0\n');
fprintf('\n');
