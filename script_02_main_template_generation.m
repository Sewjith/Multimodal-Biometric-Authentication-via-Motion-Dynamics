% Generate Reference & Testing Templates
clear; 
close all; 
clc;

% Parameters with validation
feature_dir = 'user_features';
users = {'U1', 'U2', 'U3', 'U4', 'U5', 'U6', 'U7', 'U8', 'U9', 'U10'};
num_users = length(users);

fprintf('Loading features from individual user files...\n');

fd_features_cell = cell(num_users, 1);
md_features_cell = cell(num_users, 1);
fd_users_cell = cell(num_users, 1);
md_users_cell = cell(num_users, 1);

valid_users = [];
user_sample_counts = zeros(num_users, 2); % 

for user_idx = 1:num_users
    filename = fullfile(feature_dir, sprintf('%s_features.csv', users{user_idx}));
    
    if exist(filename, 'file')
        fprintf('Loading features for %s...\n', users{user_idx});
        
        try
            user_data = csvread(filename);
            
            if size(user_data, 2) < 4
                fprintf('Warning: Insufficient columns in %s. Skipping.\n', filename);
                continue;
            end
            
            user_ids = user_data(:, 1);
            day_flags = user_data(:, 2);
            features = user_data(:, 4:end);
            
            fd_indices = day_flags == 1;
            md_indices = day_flags == 2;
            
            fd_features_user = features(fd_indices, :);
            md_features_user = features(md_indices, :);
            
            fd_features_cell{user_idx} = fd_features_user;
            md_features_cell{user_idx} = md_features_user;
            fd_users_cell{user_idx} = user_idx * ones(size(fd_features_user, 1), 1);
            md_users_cell{user_idx} = user_idx * ones(size(md_features_user, 1), 1);
            
            user_sample_counts(user_idx, :) = [size(fd_features_user, 1), size(md_features_user, 1)];
            valid_users = [valid_users, user_idx];
            
            fprintf('  %s: %d FD samples, %d MD samples\n', users{user_idx}, ...
                    size(fd_features_user, 1), size(md_features_user, 1));
                    
        catch ME
            fprintf('Error loading %s: %s\n', filename, ME.message);
            continue;
        end
    else
        fprintf('Warning: Feature file not found for %s\n', users{user_idx});
    end
end

% Combine all data
all_fd_features = vertcat(fd_features_cell{valid_users});
all_md_features = vertcat(md_features_cell{valid_users});
all_fd_users = vertcat(fd_users_cell{valid_users});
all_md_users = vertcat(md_users_cell{valid_users});

fprintf('\nTotal FD (Training) samples: %d\n', size(all_fd_features, 1));
fprintf('Total MD (Testing) samples: %d\n', size(all_md_features, 1));
fprintf('Feature dimension: %d\n', size(all_fd_features, 2));

fprintf('\nPerforming feature selection...\n');

% Combine all features for selection
all_features_combined = [all_fd_features; all_md_features];
num_features = size(all_features_combined, 2);

% Variance-based selection
feature_variance = var(all_features_combined);
[~, variance_indices] = sort(feature_variance, 'descend');

num_selected = min(50, floor(num_features * 0.3)); 
selected_features_indices = variance_indices(1:num_selected);

fprintf('Selected top %d features from %d total features (%.1f%% reduction)\n', ...
        num_selected, num_features, (1 - num_selected/num_features) * 100);

% Apply feature selection
fd_features_selected = all_fd_features(:, selected_features_indices);
md_features_selected = all_md_features(:, selected_features_indices);

%Create reference templates with validation
fprintf('\nCreating reference templates for each user...\n');

reference_templates = zeros(num_users, num_selected);
user_training_samples = zeros(num_users, 1);
user_has_data = false(num_users, 1);

for user = 1:num_users
    user_indices = all_fd_users == user;
    user_features = fd_features_selected(user_indices, :);
    
    if ~isempty(user_features) && size(user_features, 1) >= 3 
        reference_templates(user, :) = mean(user_features, 1, 'omitnan');
        user_training_samples(user) = size(user_features, 1);
        user_has_data(user) = true;
    else
        fprintf('Warning: Insufficient FD data for user %d (%s)\n', user, users{user});
        reference_templates(user, :) = NaN(1, num_selected);
        user_has_data(user) = false;
    end
end

fprintf('Reference templates created for %d/%d users\n', sum(user_has_data), num_users);

%Enhanced user templates with additional statistics
fprintf('\nCreating detailed user templates...\n');

user_templates = struct();

for user = 1:num_users
    user_indices = all_fd_users == user;
    user_features = fd_features_selected(user_indices, :);
    
    user_templates(user).user_id = user;
    user_templates(user).username = users{user};
    user_templates(user).has_sufficient_data = false;
    user_templates(user).training_count = 0;
    
    if ~isempty(user_features) && size(user_features, 1) >= 3
        user_templates(user).reference_template = mean(user_features, 1, 'omitnan');
        user_templates(user).all_training_samples = user_features;
        user_templates(user).training_count = size(user_features, 1);
        
        if size(user_features, 1) > 1
            user_templates(user).covariance = cov(user_features, 'omitrows');
            user_templates(user).feature_std = std(user_features, 'omitnan');
            user_templates(user).feature_range = [min(user_features, [], 1); max(user_features, [], 1)];
        else
            user_templates(user).covariance = eye(num_selected);
            user_templates(user).feature_std = zeros(1, num_selected);
            user_templates(user).feature_range = [user_features; user_features];
        end
        
        user_templates(user).has_sufficient_data = true;
    else
        user_templates(user).reference_template = NaN(1, num_selected);
        user_templates(user).all_training_samples = [];
        user_templates(user).covariance = NaN(num_selected);
        user_templates(user).feature_std = NaN(1, num_selected);
        user_templates(user).feature_range = NaN(2, num_selected);
    end
end

fprintf('Preparing training and testing data...\n');
valid_train = all(~isnan(fd_features_selected), 2);
valid_test = all(~isnan(md_features_selected), 2);

X_train = fd_features_selected(valid_train, :);
y_train = all_fd_users(valid_train);
X_test = md_features_selected(valid_test, :);
y_test_actual = all_md_users(valid_test);

fprintf('After NaN removal: %d training, %d testing samples\n', ...
        size(X_train, 1), size(X_test, 1));

fprintf('Creating user-specific datasets...\n');

user_specific_data = struct();

for user = 1:num_users
    % Training data (FD) 
    user_train_indices = (all_fd_users == user) & valid_train;
    user_test_indices = (all_md_users == user) & valid_test;
    
    user_specific_data(user).X_train = X_train(user_train_indices, :);
    user_specific_data(user).y_train = y_train(user_train_indices);
    user_specific_data(user).X_test = X_test(user_test_indices, :);
    user_specific_data(user).y_test = y_test_actual(user_test_indices);
    
    user_specific_data(user).positive_samples = X_train(user_train_indices, :);
    other_users = setdiff(1:num_users, user);
    negative_indices = ismember(all_fd_users(valid_train), other_users);
    user_specific_data(user).negative_samples = X_train(negative_indices, :);
    
    user_specific_data(user).train_sample_count = size(user_specific_data(user).X_train, 1);
    user_specific_data(user).test_sample_count = size(user_specific_data(user).X_test, 1);
    user_specific_data(user).negative_sample_count = size(user_specific_data(user).negative_samples, 1);
end

%Save the prepared data with versioning
fprintf('Saving prepared data...\n');

timestamp = datestr(now, 'yyyymmdd_HHMMSS');
output_file = sprintf('prepared_data_%s.mat', timestamp);

save(output_file, ...
     'X_train', 'y_train', 'X_test', 'y_test_actual', ...
     'reference_templates', 'selected_features_indices', ...
     'all_fd_users', 'all_md_users', 'user_templates', ...
     'user_specific_data', 'users', 'user_has_data', ...
     'feature_variance', 'timestamp');

copyfile(output_file, 'prepared_data.mat');

user_template_file = sprintf('user_reference_templates_%s.mat', timestamp);
save(user_template_file, 'user_templates', 'selected_features_indices', 'users', 'timestamp');
copyfile(user_template_file, 'user_reference_templates.mat');

fprintf('\nATA PREPARATION SUMMARY\n');
fprintf('Timestamp: %s\n', timestamp);
fprintf('Total training samples (FD): %d\n', size(X_train, 1));
fprintf('Total testing samples (MD): %d\n', size(X_test, 1));
fprintf('Number of selected features: %d\n', num_selected);
fprintf('Number of users: %d\n', num_users);
fprintf('Users with sufficient data: %d\n', sum(user_has_data));

fprintf('\nUSER-WISE BREAKDOWN\n');
for user = 1:num_users
    train_count = user_specific_data(user).train_sample_count;
    test_count = user_specific_data(user).test_sample_count;
    status = 'OK';
    
    if train_count == 0
        status = 'NO TRAINING';
    elseif test_count == 0
        status = 'NO TESTING';
    elseif train_count < 5
        status = 'LOW TRAINING';
    end
    
    fprintf('User %2d (%s): %2d training, %2d testing samples [%s]\n', ...
            user, users{user}, train_count, test_count, status);
end

fprintf('\nFEATURE INFORMATION\n');
fprintf('Original feature dimension: %d\n', num_features);
fprintf('Selected feature dimension: %d\n', num_selected);
fprintf('Feature reduction: %.1f%%\n', (1 - num_selected/num_features) * 100);

fprintf('\nTop 10 most important features :\n');
top_features = selected_features_indices(1:min(10, num_selected));
for i = 1:length(top_features)
    fprintf('  Feature %4d: variance = %.4f\n', top_features(i), feature_variance(top_features(i)));
end

fprintf('\nDATA QUALITY CHECKS\n');
fprintf('Samples with NaN values removed: %d training, %d testing\n', ...
        sum(~valid_train), sum(~valid_test));
fprintf('Minimum training samples per user: %d\n', min(user_training_samples(user_has_data)));
fprintf('Maximum training samples per user: %d\n', max(user_training_samples));