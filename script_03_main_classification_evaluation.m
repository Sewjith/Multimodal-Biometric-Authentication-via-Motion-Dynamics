% This script trains and tests Feedforward Neural Network for user authentication
clear all; 
close all; 
clc;

%Load prepared data
load('prepared_data.mat');

fprintf('Training data: %d samples\n', size(X_train, 1));
fprintf('Testing data: %d samples\n', size(X_test, 1));
fprintf('Number of users: %d\n', length(unique(y_train)));

load('user_reference_templates.mat');
fprintf('Loaded templates for %d users\n', length(user_templates));

%Display dataset information
fprintf('\nDATASET INFORMATION\n');
for user = 1:length(user_templates)
    train_count = user_templates(user).training_count;
    test_count = sum(y_test_actual == user);
    fprintf('User %d (%s): %d training, %d testing samples\n', ...
            user, user_templates(user).username, train_count, test_count);
end

%Prepare labels for binary classification 
num_users = length(user_templates);
results = struct();

for current_user = 1:num_users
    fprintf('\nProcessing User %d (%s)\n', current_user, user_templates(current_user).username);
    
    if user_templates(current_user).training_count == 0
        fprintf('Warning: No training data for user %d. Skipping...\n', current_user);
        
        % Store empty results for this user
        results(current_user).user_id = current_user;
        results(current_user).username = user_templates(current_user).username;
        results(current_user).TP = 0;
        results(current_user).TN = 0;
        results(current_user).FP = 0;
        results(current_user).FN = 0;
        results(current_user).FAR = 1; 
        results(current_user).FRR = 1; 
        results(current_user).EER = 1; 
        results(current_user).accuracy = 0;
        results(current_user).has_data = false;
        continue;
    end

    % For training, label=1 for current user, label=0 for all other users
    y_train_binary = (y_train == current_user);
    
    y_test_binary_actual = (y_test_actual == current_user);
    
    %Balance the training data
    positive_samples = sum(y_train_binary);
    negative_samples = sum(~y_train_binary);
    
    fprintf('Training - Positive samples (user %d): %d\n', current_user, positive_samples);
    fprintf('Training - Negative samples (other users): %d\n', negative_samples);
    
    if positive_samples < 2
        fprintf('Warning: Insufficient positive samples for user %d. Skipping...\n', current_user);
        
        % Store empty results for this user
        results(current_user).user_id = current_user;
        results(current_user).username = user_templates(current_user).username;
        results(current_user).TP = 0;
        results(current_user).TN = 0;
        results(current_user).FP = 0;
        results(current_user).FN = 0;
        results(current_user).FAR = 1;
        results(current_user).FRR = 1;
        results(current_user).EER = 1;
        results(current_user).accuracy = 0;
        results(current_user).has_data = false;
        continue;
    end
    
    %Handle class imbalance using random undersampling
    if negative_samples > positive_samples * 3
        fprintf('Applying class balancing...\n');
        
        positive_indices = find(y_train_binary);
        negative_indices = find(~y_train_binary);
        
        max_negative_samples = min(length(negative_indices), positive_samples * 2);
        selected_negative_indices = negative_indices(randperm(length(negative_indices), max_negative_samples));
        
        balanced_indices = [positive_indices; selected_negative_indices];
        balanced_indices = balanced_indices(randperm(length(balanced_indices))); 
        
        X_train_balanced = X_train(balanced_indices, :);
        y_train_balanced = y_train_binary(balanced_indices);
        
        fprintf('Balanced dataset: %d positive, %d negative samples\n', ...
                sum(y_train_balanced), sum(~y_train_balanced));
    else
        % Use original data if not severely imbalanced
        X_train_balanced = X_train;
        y_train_balanced = y_train_binary;
    end
    
    %Feedforward Neural Network
    fprintf('Training neural network for user %d...\n', current_user);

    hidden_layer_size = min(20, floor(size(X_train_balanced, 1) / 5)); 
    hidden_layer_size = max(5, hidden_layer_size); 
    
    fprintf('Using hidden layer size: %d neurons\n', hidden_layer_size);
    
    net = feedforwardnet(hidden_layer_size);
    
    net.trainFcn = 'trainlm'; 
    net.trainParam.epochs = 100; 
    net.trainParam.goal = 1e-5; 
    net.trainParam.show = 10; 
    net.trainParam.max_fail = 10; 
    
    net.divideFcn = 'dividerand';
    if size(X_train_balanced, 1) > 50
        net.divideParam.trainRatio = 0.7;
        net.divideParam.valRatio = 0.15;
        net.divideParam.testRatio = 0.15;
    else
        net.divideParam.trainRatio = 0.8;
        net.divideParam.valRatio = 0.1;
        net.divideParam.testRatio = 0.1;
    end
    
    % Train the network
    try
        [net, tr] = train(net, X_train_balanced', y_train_balanced');
        training_successful = true;
        fprintf('Training completed for user %d\n', current_user);
    catch ME
        fprintf('Training failed for user %d: %s\n', current_user, ME.message);
        training_successful = false;
    end
    
    %Evaluate the model
    if training_successful
        fprintf('Testing neural network for user %d...\n', current_user);
        
        y_test_pred = net(X_test');
        
        threshold = 0.5;
        y_test_pred_binary = (y_test_pred > threshold);
        
        %Calculate performance metrics
        TP = sum(y_test_pred_binary & y_test_binary_actual');
        TN = sum(~y_test_pred_binary & ~y_test_binary_actual');
        FP = sum(y_test_pred_binary & ~y_test_binary_actual');
        FN = sum(~y_test_pred_binary & y_test_binary_actual');
        
        if (TP + FN) > 0
            FRR = FN / (TP + FN); 
        else
            FRR = 0;
        end
        
        if (FP + TN) > 0
            FAR = FP / (FP + TN); 
        else
            FAR = 0;
        end
        
        % Calculate Equal Error Rate 
        EER = (FAR + FRR) / 2;
        
        accuracy = (TP + TN) / (TP + TN + FP + FN);
        
        results(current_user).training_samples = size(X_train_balanced, 1);
        results(current_user).hidden_layer_size = hidden_layer_size;
    else
        TP = 0; TN = 0; FP = 0; FN = 0;
        FAR = 1; FRR = 1; EER = 1; accuracy = 0;
    end
    
    results(current_user).user_id = current_user;
    results(current_user).username = user_templates(current_user).username;
    results(current_user).TP = TP;
    results(current_user).TN = TN;
    results(current_user).FP = FP;
    results(current_user).FN = FN;
    results(current_user).FAR = FAR;
    results(current_user).FRR = FRR;
    results(current_user).EER = EER;
    results(current_user).accuracy = accuracy;
    results(current_user).has_data = training_successful;
    
    if training_successful
        results(current_user).network = net;
    end
    
    fprintf('User %d (%s) Results:\n', current_user, user_templates(current_user).username);
    fprintf('  TP: %d, TN: %d, FP: %d, FN: %d\n', TP, TN, FP, FN);
    fprintf('  FAR: %.4f, FRR: %.4f, EER: %.4f\n', FAR, FRR, EER);
    fprintf('  Accuracy: %.4f\n', accuracy);
    
end 

%% Calculate overall performance 
fprintf('\nOVERALL RESULTS\n');

users_with_data = [results.has_data];
if any(users_with_data)
    valid_results = results(users_with_data);
    
    overall_FAR = mean([valid_results.FAR]);
    overall_FRR = mean([valid_results.FRR]);
    overall_EER = mean([valid_results.EER]);
    overall_accuracy = mean([valid_results.accuracy]);
    
    fprintf('Overall FAR: %.4f (based on %d users)\n', overall_FAR, sum(users_with_data));
    fprintf('Overall FRR: %.4f (based on %d users)\n', overall_FRR, sum(users_with_data));
    fprintf('Overall EER: %.4f (based on %d users)\n', overall_EER, sum(users_with_data));
    fprintf('Overall Accuracy: %.4f (based on %d users)\n', overall_accuracy, sum(users_with_data));
else
    fprintf('No users with sufficient data for analysis\n');
    overall_FAR = 1;
    overall_FRR = 1;
    overall_EER = 1;
    overall_accuracy = 0;
end

%% Display detailed results table
fprintf('\nDETAILED RESULTS FOR ALL USERS\n');
fprintf('User\tUsername\tTrainSamples\tTP\tTN\tFP\tFN\tFAR\t\tFRR\t\tEER\t\tAccuracy\n');
fprintf('----\t--------\t-----------\t--\t--\t--\t--\t-----\t\t-----\t\t-----\t\t--------\n');

for user = 1:num_users
    if results(user).has_data
        train_samples = results(user).training_samples;
    else
        train_samples = 0;
    end
    
    fprintf('%d\t%s\t\t%d\t\t%d\t%d\t%d\t%d\t%.4f\t%.4f\t%.4f\t%.4f', ...
            results(user).user_id, ...
            results(user).username, ...
            train_samples, ...
            results(user).TP, ...
            results(user).TN, ...
            results(user).FP, ...
            results(user).FN, ...
            results(user).FAR, ...
            results(user).FRR, ...
            results(user).EER, ...
            results(user).accuracy);
    
    if ~results(user).has_data
        fprintf(' *');
    end
    fprintf('\n');
end

%Save results
save('classification_results.mat', 'results', 'overall_FAR', 'overall_FRR', 'overall_EER', 'overall_accuracy', 'users_with_data');
fprintf('Summary: %d out of %d users had sufficient data for analysis\n', sum(users_with_data), num_users);