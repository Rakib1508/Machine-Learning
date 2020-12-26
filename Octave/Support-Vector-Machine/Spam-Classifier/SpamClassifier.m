%% In this project, we will be implementing SVM model in order to classify
%% emails and sort out the spam mails. We will use a short word list and search
%% mail contents for availability of those words and perform binary
%% classification. We also perform stemming over the mail contents.

% Clean up space for performing classification
clear;
close all;
clc;

%% Step 1: Email content preprocessing
fprintf('Pre-processing sample email content....\n\n');
pause(1);

% extract features from mail contents
mail_contents = FileScanner('Sample1.txt');
word_indices = MailContentProcessor(mail_contents);

% Print Stats
fprintf('Word Indices: \n');
fprintf(' %d', word_indices);
fprintf('\n\n');

fprintf('Mail preprocessed. Press enter to extract features....\n\n');
pause;


%% Step 2: Feature extraction
fprintf('Extracting features from email content....\n\n');
pause(1);

% Extract features from indices
features = FeatureExtractor(word_indices);

% Print Stats
fprintf('Length of feature vector: %d\n', length(features));
fprintf('Number of non-zero entries: %d\n', sum(features > 0));

fprintf('Program paused. Press enter to continue....\n\n');
pause;


%% Step 3: Train linear SVM for spam classifier
fprintf('Training Linear SVM for Spam Classification....\n');
fprintf('(this may take 1 to 2 minutes) ...\n\n');
pause(1);

% Load spam email training data
load('TrainData.mat');

% Initialize required variables
C = 0.1;
model = SvmTrainer(X, y, C, @LinearKernel);
prediction = SvmPredict(model, X);

fprintf('Training Accuracy: %f\n', mean(double(prediction == y)) * 100);

fprintf('Program paused. Press enter to continue....\n\n');
pause;


%% Step 4: Test Spam classification
fprintf('Evaluating the trained Linear SVM on a test set....\n\n');
pause(1);

load('TestData.mat');

prediction = SvmPredict(model, Xtest);
fprintf('Test Accuracy: %f\n', mean(double(prediction == ytest)) * 100);

fprintf('Program paused. Press enter to continue....\n\n');
pause;


%% Step 5: Spam prediction system setup
fprintf('Running spam classifier....\n\n');
pause(1);

[weight, idx] = sort(model.w, 'descend');
vocabList = VocabGenerator();

fprintf('\nTop predictors of spam: \n');
for i = 1:15
    fprintf(' %-15s (%f) \n', vocabList{idx(i)}, weight(i));
end

fprintf('\n\n');
fprintf('Program paused. Press enter to continue....\n\n');
pause;


%% Step 6: Try real mail to classify
fprintf('Implementing spam classifier....\n\n');
pause(1);

filename = 'SpamSample1.txt';
    
file_contents = FileScanner(filename);
word_indices = MailContentProcessor(file_contents);
x = FeatureExtractor(word_indices);

prediction = SvmPredict(model, x);
fprintf('Processed %s\n\nSpam Classification: %d\n', filename, prediction);
fprintf('(1 indicates spam, 0 indicates not spam)\n\n');

filename = 'SpamSample2.txt';
    
file_contents = FileScanner(filename);
word_indices = MailContentProcessor(file_contents);
x = FeatureExtractor(word_indices);

prediction = SvmPredict(model, x);
fprintf('Processed %s\n\nSpam Classification: %d\n', filename, prediction);
fprintf('(1 indicates spam, 0 indicates not spam)\n\n');
