%% Extra Exercise: PCA for Visualization
%% One useful application of PCA is to use it to visualize high-dimensional
%% data. In the last K-Means exercise you ran K-Means on 3-dimensional 
%% pixel colors of an image. We first visualize this output in 3D, and then
%% apply PCA to obtain a visualization in 2D.

% Clean up space for running PCA algorithm
close all;
close all;
clc;

% Reload the image from the previous exercise and run K-Means on it
A = double(imread('SmallBird.png'));

A = A / 255;
img_size = size(A);
X = reshape(A, img_size(1) * img_size(2), 3);
K = 16; 
max_iters = 10;

initial_centroids = CentroidsInitializer(X, K);
[centroids, idx] = KmeansClusterer(X, initial_centroids, max_iters);

sel = floor(rand(1000, 1) * size(X, 1)) + 1;

%  Setup Color Palette
palette = hsv(K);
colors = palette(idx(sel), :);

%  Visualize the data and centroid memberships in 3D
figure;
scatter3(X(sel, 1), X(sel, 2), X(sel, 3), 10, colors);
title('Pixel dataset plotted in 3D. Color shows centroid memberships');
fprintf('Program paused. Press enter to continue.\n');
pause;

%% Extra Exercise: PCA for Visualization
%% Use PCA to project this cloud to 2D for visualization

% Subtract the mean to use PCA
[X_norm, mu, sigma] = Normalizer(X);

% PCA and project the data to 2D
[U, S] = PCA(X_norm);
Z = ProjectData(X_norm, U, 2);

% Plot in 2D
figure;
PlotDataPoints(Z(sel, :), idx(sel), K);
title('Pixel dataset plotted in 2D, using PCA for dimensionality reduction');
