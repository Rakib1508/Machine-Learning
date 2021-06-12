%% We are going to implement binary classification using SVM algorithm
%% in this project. We will also separate training, validation and test sets.
%% We will implement Gaussian Kernel in order to perform SVM over our
%% training set examples.

% Clean up space before start learning
clear;
close all;
clc;

%% Step 1: Loading and visualizing data
fprintf('Loading data from warming up set....\n\n');
pause(1);

% Load data for warmup
load('WarmupTrainingSet.mat');

fprintf('Loading complete. Press enter to visualize....\n\n');
pause;

fprintf('Visualizing data....\n\n');
pause(1);

% Visualize data on a graph
DataPlotter(X, y);

fprintf('Data plotted. Press enter to run linear kernel....\n\n');
pause;


%% Step 2: Train Linear SVM
fprintf('Training Linear SVM....\n');
pause(1);

% initialize environment
C = 1;
model = SvmTrainer(X, y, C, @LinearKernel, 1e-3, 20);
VisualizeLinearBoundary(X, y, model);

fprintf('Linear SVM complete. Press enter to run gaussian kernel....\n\n');
pause;


%% Step 3: Implement Gaussian Kernel
fprintf('Evaluating Gaussian Kernel....\n\n');
pause(1);

% Initialize required variables
x1 = [1 2 1];
x2 = [0 4 -1];
sigma = 2;

% Perform gaussian kernel
sim = GaussianKernel(x1, x2, sigma);

fprintf(['Gaussian Kernel between x1 = [1; 2; 1], x2 = [0; 4; -1], sigma = %f :' ...
         '\n\t%f\n(for sigma = 2, this value should be about 0.324652)\n\n'], sigma, sim);

fprintf('Gaussian Kernel evaluated. Press enter to visualize dataset....\n\n');
pause;


%% Step 4: Load and Visualize Training set
fprintf('Loading data from training set....\n\n');
pause(1);

% Load data for warmup
load('TrainingSet1.mat');

fprintf('Loading complete. Press enter to visualize....\n\n');
pause;

fprintf('Visualizing data....\n\n');
pause(1);

% Visualize data on a graph
DataPlotter(X, y);

fprintf('Data plotted. Press enter to run RBF kernel....\n\n');
pause;


%% Step 5: Train data with RBF Kernel
fprintf('Evaluating RBF Kernel (this may take a while)....\n\n');
pause(1);

% Initialize required variables
C = 1;
sigma = 0.1;

% We set the tolerance and max_passes lower here so that the code will run
% faster. However, in practice, you will want to run the training to convergence.
model = SvmTrainer(X, y, C, @(x1, x2) GaussianKernel(x1, x2, sigma));
VisualizeDecisionBoundary(X, y, model);

fprintf('RFB kernel implemented. Press enter to work new dataset....\n\n');
pause;


%% Step 6: Load and Visualize more complex dataset
fprintf('Loading data from training set....\n\n');
pause(1);

% Load data for warmup
load('ComplexTrainingData.mat');

fprintf('Loading complete. Press enter to visualize....\n\n');
pause;

fprintf('Visualizing data....\n\n');
pause(1);

% Visualize data on a graph
DataPlotter(X, y);

fprintf('Data plotted. Press enter to run RBF kernel....\n\n');
pause;


%% Step 7: Train complex data with RBF Kernel
fprintf('Evaluating RBF Kernel (this may take a while)....\n\n');
pause(1);

% Initialize required variables
[C, sigma] = DatasetParameters(X, y, Xval, yval);

% We set the tolerance and max_passes lower here so that the code will run
% faster. However, in practice, you will want to run the training to convergence.
model = SvmTrainer(X, y, C, @(x1, x2) GaussianKernel(x1, x2, sigma));
VisualizeDecisionBoundary(X, y, model);

fprintf('Obtained optimized parameters are: \n\n');
fprintf('C = %f\t sigma = %f\n', C, sigma);
