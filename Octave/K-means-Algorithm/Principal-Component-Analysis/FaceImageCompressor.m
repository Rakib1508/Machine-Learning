%% In this project, we will use principle component analysis, a.k.a. PCA
%% in order to reduce the dimensions of a set of facial images.

% clean up space to initialize learning algorithm
clear;
close all;
clc;

%% Step 1: Load and visualize example dataset
fprintf('Visualizing example dataset for PCA....\n\n');
pause(1);

% load data from the dataset
load('ExampleData.mat');

% visualize example dataset
plot(X(:, 1), X(:, 2), 'bo');
axis([0.5 6.5 2 8]);
axis square;

fprintf('Data loaded. Press enter to continue....\n\n');
pause;


%% Step 2: Principle Component Analysis
fprintf('Running PCA on example data....\n\n');
pause(1);

% Must normalize before running PCA
[X_norm, mu, sigma] = Normalizer(X);
[U, S] = PCA(X_norm);

% Compute mu, the mean of the each feature
% Draw the eigenvectors centered at mean of data. These lines show the
% directions of maximum variations in the dataset.
hold on;
DrawLine(mu, mu + 1.5 * S(1,1) * U(:,1)', '-k', 'LineWidth', 2);
DrawLine(mu, mu + 1.5 * S(2,2) * U(:,2)', '-k', 'LineWidth', 2);
hold off;

fprintf('Top eigenvector: \n');
fprintf(' U(:,1) = %f %f \n', U(1,1), U(2,1));
fprintf('\n(you should expect to see -0.707107 -0.707107)\n\n');

fprintf('PCA complete. Press enter to continue....\n\n');
pause;


%% Step 3: Dimension reduction
fprintf('Executing dimension reduction on example dataset....\n\n');
pause(1);

% Plot normalized dataset
plot(X_norm(:, 1), X_norm(:, 2), 'bo');
axis([-4 3 -4 3]);
axis square;

% Project data with K = 1
K = 1;
Z = ProjectData(X_norm, U, K);
fprintf('Projection of the first example: %f\n', Z(1));
fprintf('\n(this value should be about 1.481274)\n\n');

X_rec = RecoverData(Z, U, K);
fprintf('Approximation of the first example: %f %f\n', X_rec(1, 1), X_rec(1, 2));
fprintf('\n(this value should be about  -1.047419 -1.047419)\n\n');

%  Draw lines connecting the projected points to the original points
hold on;
plot(X_rec(:, 1), X_rec(:, 2), 'ro');

for i = 1:size(X_norm, 1),
    DrawLine(X_norm(i, :), X_rec(i, :), '--k', 'LineWidth', 1);
endfor

hold off;

fprintf('Program paused. Press enter to continue....\n\n');
pause;


%% Step 4: Loading and Visualizing face data
fprintf('Loading face dataset....\n\n');
pause(1);

% load face data
load('FaceData.mat');

% display first 100 face images
DisplayData(X(1:100, :));

fprintf('Data loaded. Press enter to continue....\n\n');
pause;


%% Step 5: PCA on face data - Eigenfaces
fprintf(['\nRunning PCA on face dataset....\n' ...
         '(this might take a minute or two ....)\n\n']);

% must normalize before running PCA
[X_norm, mu, sigma] = Normalizer(X);

% Run PCA algorithm
[U, S] = PCA(X_norm);

% Visualize top 36 eigenvectors found
DisplayData(U(:, 1:36)');

fprintf('PCA completed. Press enter to continue....\n\n');
pause;


%% Step 6: Dimension reduction for face data
fprintf('Dimension reduction for face dataset....\n\n');
pause(1);

% initialize required variables
K = 100;
Z = ProjectData(X_norm, U, K);

fprintf('The projected data Z has a size of: ')
fprintf('%d \n\n', size(Z));

fprintf('Dimensions reduced. Press enter to continue....\n\n');
pause;


%% Step 7: Visualize face data after PCA Dimension reduction
fprintf('Visualizing the projected (reduced dimension) faces....\n\n');
pause(1);

X_rec = RecoverData(Z, U, K);

% Display normalized data
subplot(1, 2, 1);
DisplayData(X_norm(1:100, :));
title('Original faces');
axis square;

% Display reconstructed data from only k eigenfaces
subplot(1, 2, 2);
DisplayData(X_rec(1:100,:));
title('Recovered faces');
axis square;
