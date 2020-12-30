%% In this project, we are going to compress images using K-means algorithm
%% by reducing the number of distinct colors. In the name of the algorithm,
%% K refers to the number of distinct colors on the image.

% Clean up spaces for initializing training process
clear;
close all;
clc;

%% Step 1: Find closest centroids
fprintf('Finding closest centroids....\n\n');
pause(1);

% Load an example dataset
load('TrainingExample.mat');

% Initialize required variables
K = 3;
initial_centroids = [3 3; 6 2; 8 5];

% find the closest centroids starting with initial assignments
idx = FindClosestCentroids(X, initial_centroids);

fprintf('Closest centroids for the first 3 examples: \n')
fprintf(' %d\n', idx(1:3));
fprintf('(the closest centroids should be 1, 3, 2 respectively)\n\n');

fprintf('Closest centriods found. Press enter to compute means....\n\n');
pause;


%% Step 2: Compute means
fprintf('Computing centroid means....\n\n');
pause(1);

% Compute centroids based on the closest centroids found previously
centroids = ComputeCentroids(X, idx, K);

fprintf('Centroids computed after initial finding of closest centroids: \n')
fprintf(' %f %f \n' , centroids');
fprintf('\n(the centroids should be\n');
fprintf('   [ 2.428301 3.157924 ]\n');
fprintf('   [ 5.813503 2.633656 ]\n');
fprintf('   [ 7.119387 3.616684 ]\n\n');

fprintf('Centroids computed. Press enter to start clustering....\n\n');
pause;


%% Step 3: K-means clustering
fprintf('Running K-means clustering....\n\n');
pause(1);

% initialize required variables
K = 3;
max_iters = 10;

[centroids, idx] = KmeansClusterer(X, initial_centroids, max_iters, true);

fprintf('\nK-means completed. Press enter to cluster on pixels....\n\n');
pause;


%% Step 4: K-Means Clustering on Pixels
fprintf('Running K-Means clustering on pixels from an image....\n\n');
pause(1);

% load an image
A = double(imread('SmallBird.png'));

% divide by 255 to normalize all features within a range of 0 to 1
A = A / 255;

% initialize required variables
img_size = size(A);
X = reshape(A, img_size(1) * img_size(2), 3);
K = 16;
max_iters = 10;

% initialize centroids
initial_centroids = CentroidsInitializer(X, K);

% perform K-means
[centroids, idx] = KmeansClusterer(X, initial_centroids, max_iters);

fprintf('\nK-means completed. Press enter to continue....\n\n');
pause;


%% Step 5: Image compression
fprintf('Applying K-Means to compress an image....\n\n');
pause(1);

% find closest cluster data points
idx = FindClosestCentroids(X, centroids);

% We can now recover the image from the indices (idx) by mapping each pixel
% (specified by its index in idx) to the centroid value
X_recovered = centroids(idx, :);

% Reshape the recovered image into proper dimensions
X_recovered = reshape(X_recovered, img_size(1), img_size(2), 3);

% Display original image
subplot(1, 2, 1);
imagesc(A);
title('Original image');

% Display compressed image side by side
subplot(1, 2, 2);
imagesc(X_recovered)
title(sprintf('Compressed, with %d colors.', K));
