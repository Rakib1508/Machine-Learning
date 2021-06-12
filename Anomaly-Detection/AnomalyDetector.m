%% In this project, we are going to detect anomaly from the behavior of
%% network traffic through server computers. The reason for not using classic
%% logistic regression is that, the training examples are mostly non-anomalous
%% where as anomalous examples are very rare.

% Clean up space for initializing training
clear;
close all;
clc;

%% Step 1: Load and visualize data from the dataset
fprintf('Loading and visualizing data for outliner detection....\n\n');
pause(1);

% load training examples from the dataset
load('ServerTrafficData.mat');

% Visualize example dataset
plot(X(:, 1), X(:, 2), 'bx');
axis([0 30 0 30]);
xlabel('Latency (ms)');
ylabel('Throughput (mb/s)');

fprintf('Data loaded. Press enter to fit Gaussian parameters....\n\n');
pause;


%% Step 2: Estimate dataset statistics
fprintf('Visualizing Gaussian fit....\n\n');
pause(1);

% estimate mu, sigma and p
[mu sigma] = EstimateGaussianParameters(X);
p = MultivariateGaussian(X, mu, sigma);

% Visualize the fit
VisualizeGaussianFit(X, mu, sigma);
xlabel('Latency (ms)');
ylabel('Throughput (mb/s)');

fprintf('Gaussian fit visualized. Press enter to continue....\n\n');
pause;


%% Step 3: Find outliers
fprintf('Calculating threshold....\n\n');

% threshold calculation
pval = MultivariateGaussian(Xval, mu, sigma);
[epsilon F1] = ThresholdFinder(yval, pval);

fprintf('Best epsilon found using cross-validation: %e\n', epsilon);
fprintf('Best F1 on Cross Validation Set:  %f\n', F1);
fprintf('   (you should see a value epsilon of about 8.99e-05)\n');
fprintf('   (you should see a Best F1 value of  0.875000)\n\n');

%  Find the outliers in the training set and plot it
outliers = find(p < epsilon);

% draw red circle around outliers
hold on;
plot(X(outliers, 1), X(outliers, 2), 'ro', 'LineWidth', 2, 'MarkerSize', 10);
hold off;

fprintf('Thresold found. Press enter to continue....\n\n');
pause;


%% Step 4: Perform cross validation
fprintf('Loading Cross-validation set....\n\n');
pause(1);

% load cross-validation set
load('CrossValidationData.mat');

% perform learning on training set
[mu sigma] = EstimateGaussianParameters(X);
p = MultivariateGaussian(X, mu, sigma);

% perform cross-validation
pval = MultivariateGaussian(Xval, mu, sigma);

% Find the best threshold
[epsilon F1] = ThresholdFinder(yval, pval);

fprintf('Best epsilon found using cross-validation: %e\n', epsilon);
fprintf('Best F1 on Cross Validation Set:  %f\n', F1);
fprintf('   (you should see a value epsilon of about 1.38e-18)\n');
fprintf('   (you should see a Best F1 value of 0.615385)\n');
fprintf('# Outliers found: %d\n\n', sum(p < epsilon));
