%% In this project, we will make a movie recommender system based on user
%% reviews using Collaborative Feature learning algorithm. The algorithm
%% will learn features by itself even when there is a new feature emerges
%% down the time dynamically.

% Clean up space for training the system
clear;
close all;
clc;

%% Step 1: Loading and visualizing movie ratings
fprintf('Loading movie ratings....\n\n');
pause(1);

% load training data from dataset
load('TrainingData.mat');

% Y is a 1682x943 matrix, containing ratings (1-5) of 1682 movies on 943 users
% R is a 1682x943 matrix, R(i,j) = 1 only if user j gave a rating to movie i

%  From the matrix, we can compute statistics like average rating.
fprintf('Average rating for movie 1 (Toy Story): %f / 5\n\n', ...
        mean(Y(1, R(1, :))));

%  We can "visualize" the ratings matrix by plotting it with imagesc
imagesc(Y);
ylabel('Movies');
xlabel('Users');

fprintf('Data loaded. Press enter to continue....\n\n');
pause;


%% Step 2: Collaborative filtering cost function
fprintf('Computing collaborative cost....\n\n');
pause(1);

%  Load pre-trained weights (X, Theta, num_users, num_movies, num_features)
load('WeightsData.mat');

% reduce dataset size to improve computation time
num_users = 4;
num_movies = 5;
num_features = 3;
X = X(1:num_movies, 1:num_features);
Theta = Theta(1:num_users, 1:num_features);
Y = Y(1:num_movies, 1:num_users);
R = R(1:num_movies, 1:num_users);

% compute cost function
J = CostFunction([X(:); Theta(:)], Y, R, num_users, num_movies, num_features, 0);

fprintf(['Cost at loaded parameters: %f '...
         '\n(this value should be about 22.22)\n\n'], J);

fprintf('Cost computed. Press enter to continue....\n\n');
pause;


%% Step 3: Collaborative filtering gradient
fprintf('Checking Gradients (without regularization)....\n\n');
pause(1);

%  Check gradients by running CheckCostGradients
CheckCostGradients;

fprintf('Gradient checked Ok. Press enter to continue....\n\n');
pause;


%% Step 4: Collaborative filtering cost regularization
fprintf('Regularizing filtering cost....\n\n');
pause(1);

% evaluate cost value
J = CostFunction([X(:); Theta(:)], Y, R, num_users, num_movies, ...
                    num_features, 1.5);

fprintf(['Cost at loaded parameters (lambda = 1.5): %f '...
         '\n(this value should be about 31.34)\n\n'], J);

fprintf('Cost computed. Press enter to continue....\n\n');
pause;


%% Step 5: Collaborative filtering gradients regularization
fprintf('Checking gradients with regularization....\n\n');
pause(1);

% Check gradients by running CheckCostGradients
CheckCostGradients(1.5);

fprintf('Gradient checked Ok. Press enter to continue....\n\n');
pause;


%% Step 6: Input ratings from new user
fprintf('Taking user reviews....\n\n');
pause(1);

movieList = LoadMovies();

% initialize user ratings
ratings = zeros(1682, 1);

% Check the file Movie_IDs.txt for id of each movie in our dataset
% For example, Toy Story (1995) has ID 1, so to rate it "4"
ratings(1) = 4;

% Or suppose did not enjoy Silence of the Lambs (1991)
ratings(98) = 2;

% select a few movies and give some ratings
ratings(7) = 3;
ratings(12)= 5;
ratings(54) = 4;
ratings(64)= 5;
ratings(66)= 3;
ratings(69) = 5;
ratings(183) = 4;
ratings(226) = 5;
ratings(355)= 5;

fprintf('New user ratings:\n');
for i = 1:length(ratings),
    if ratings(i) > 0,
        fprintf('Rated %d for %s\n', ratings(i), movieList{i});
    endif
endfor

fprintf('\nNew user rating assigned. Press enter to continue....\n\n');
pause;


%% Step 7: Learning movie ratings
fprintf('Training collaborative filtering....\n\n');
pause(1);

% load training data from dataset
load('TrainingData.mat');

% Y is a 1682x943 matrix, containing ratings (1-5) of 1682 movies on 943 users
% R is a 1682x943 matrix, R(i,j) = 1 only if user j gave a rating to movie i

%  Add new user ratings to the data matrix
Y = [ratings Y];
R = [(ratings ~= 0) R];

% Normalize ratings
[Ynorm, Ymean] = Normalizer(Y, R);

% initialize required variables
num_users = size(Y, 2);
num_movies = size(Y, 1);
num_features = 10;

% Set Initial Parameters (Theta, X)
X = randn(num_movies, num_features);
Theta = randn(num_users, num_features);
initial_parameters = [X(:); Theta(:)];

% Set options for fmincg
options = optimset('GradObj', 'on', 'MaxIter', 100);

% Set Regularization
lambda = 10;
theta = fmincg (@(t)(CostFunction(t, Ynorm, R, num_users, num_movies, ...
                                num_features, lambda)), ...
                initial_parameters, options);

% Unfold the returned theta back into U and W
X = reshape(theta(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(theta(num_movies*num_features+1:end), ...
                num_users, num_features);

fprintf('Recommender system learning done. Press enter to continue....\n\n');
pause;


%% Step 8: Make recommendations to users
fprintf('Making recommendations....\n\n');
pause(1);

% After training, make recommendations by computing the predictions matrix.
p = X * Theta';
predictions = p(:,1) + Ymean;

movieList = LoadMovies();

[r, ix] = sort(predictions, 'descend');
fprintf('Top recommendations for you:\n');

for i=1:10,
    j = ix(i);
    fprintf('Predicting rating %.1f for movie %s\n', predictions(j), ...
            movieList{j});
endfor

fprintf('\n\nOriginal ratings provided:\n');
for i = 1:length(ratings),
    if ratings(i) > 0,
        fprintf('Rated %d for %s\n', ratings(i), movieList{i});
    endif
endfor
