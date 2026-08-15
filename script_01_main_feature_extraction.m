% Feature extraction script
clear all; 
close all; 
clc;

%Define parameters
segment_duration = 5; 
sampling_rate = 30; 
segment_length = segment_duration * sampling_rate; % samples per segment


% List of users
users = {'U1', 'U2', 'U3', 'U4', 'U5', 'U6', 'U7', 'U8', 'U9', 'U10'};
days = {'FD', 'MD'};

all_features = [];

%Process each user and each day
for user_idx = 1:length(users)
    for day_idx = 1:length(days)
        filename = [users{user_idx} 'NW_' days{day_idx} '.csv'];
        fprintf('Processing file: %s\n', filename);
        
        % Load data
        try
            data = csvread(filename);
        catch
            fprintf('Warning: Could not load file %s\n', filename);
            continue;
        end
        
        %Check if data is loaded correctly
        if isempty(data)
            fprintf('Warning: No data in file %s\n', filename);
            continue;
        end
        
        % Extract columns Time, AccX, AccY, AccZ, GyroX, GyroY, GyroZ
        time = data(:, 1);
        accX = data(:, 2);
        accY = data(:, 3);
        accZ = data(:, 4);
        gyroX = data(:, 5);
        gyroY = data(:, 6);
        gyroZ = data(:, 7);
        
        %Preprocessing
        % Apply simple moving average filter
        accX_filtered = movmean(accX, 3);
        accY_filtered = movmean(accY, 3);
        accZ_filtered = movmean(accZ, 3);
        gyroX_filtered = movmean(gyroX, 3);
        gyroY_filtered = movmean(gyroY, 3);
        gyroZ_filtered = movmean(gyroZ, 3);
        
        %Create segments
        total_samples = length(accX_filtered);
        num_segments = floor(total_samples / segment_length);
        fprintf('  Total samples: %d, Number of segments: %d\n', total_samples, num_segments);
        
        %Process each segment
        for seg = 1:num_segments
            start_idx = (seg - 1) * segment_length + 1;
            end_idx = seg * segment_length;
            
            % Extract segment data
            seg_accX = accX_filtered(start_idx:end_idx);
            seg_accY = accY_filtered(start_idx:end_idx);
            seg_accZ = accZ_filtered(start_idx:end_idx);
            seg_gyroX = gyroX_filtered(start_idx:end_idx);
            seg_gyroY = gyroY_filtered(start_idx:end_idx);
            seg_gyroZ = gyroZ_filtered(start_idx:end_idx);
            

            %Feature extraction
            features = [];
            
            % Extract features for each signal
            signals = {seg_accX, seg_accY, seg_accZ, seg_gyroX, seg_gyroY, seg_gyroZ};
            signal_names = {'AccX', 'AccY', 'AccZ', 'GyroX', 'GyroY', 'GyroZ'};
            
            for sig_idx = 1:length(signals)
                current_signal = signals{sig_idx};
                
                % Time domain features
                f1 = mean(current_signal);
                f2 = std(current_signal);
                f3 = var(current_signal);
                f4 = sqrt(mean(current_signal.^2));
                f5 = max(current_signal);
                f6 = min(current_signal);
                f7 = f5 - f6;
                f8 = median(current_signal);
                f9 = mean(abs(current_signal - f1));
                f10 = sum(current_signal.^2);
                
                % Frequency domain features
                N = length(current_signal);
                fft_signal = fft(current_signal);
                fft_magnitude = abs(fft_signal(1:floor(N/2)+1));
                frequencies = (0:floor(N/2)) * sampling_rate / N;
                f11 = sum(frequencies(1:length(fft_magnitude)) .* fft_magnitude') / sum(fft_magnitude);
                f12 = sqrt(sum(((frequencies(1:length(fft_magnitude)) - f11).^2 .* fft_magnitude')) / sum(fft_magnitude));
                f13 = sum(fft_magnitude.^2);
                pdf = fft_magnitude / sum(fft_magnitude);
                f14 = -sum(pdf .* log2(pdf + eps));
                f15 = max(fft_magnitude); 
                f16 = mean(fft_magnitude); 
                f17 = std(fft_magnitude); 
                f18 = sum(fft_magnitude > mean(fft_magnitude)); 
                f19 = f15 / f16; 
                f20 = f11 * f12;
                
                % Combine all 20 features for this signal
                signal_features = [f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, ...
                                  f11, f12, f13, f14, f15, f16, f17, f18, f19, f20];
                
                features = [features, signal_features];
            end
            
            user_id = user_idx;
            day_flag = day_idx; 
            
            feature_row = [user_id, day_flag, features];
            all_features = [all_features; feature_row];
        end 
    end 
end 

%% Save features to file
fprintf('\nFeature Extraction Summary\n');
fprintf('Total features extracted: %d\n', size(all_features, 1));
fprintf('Feature vector length: %d\n', size(all_features, 2) - 2); 
csvwrite('extracted_features.csv', all_features);
fprintf('Features saved to extracted_features.csv\n');

% Create directory for individual user features
output_dir = 'user_features';
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
    fprintf('\nCreated directory: %s\n', output_dir);
end

fprintf('\nSaving individual user files\n');
for user = 1:length(users)
    user_indices = all_features(:, 1) == user;
    user_features = all_features(user_indices, :);
    
    if isempty(user_features)
        fprintf('Warning: No features found for %s\n', users{user});
        continue;
    end
    
    num_user_samples = size(user_features, 1);
    segment_ids = (1:num_user_samples)';
    
    user_data = [user_features(:, 1:2), segment_ids, user_features(:, 3:end)];
    
    % Save to individual user file
    filename = fullfile(output_dir, sprintf('%s_features.csv', users{user}));
    csvwrite(filename, user_data);
    
    fd_count = sum(user_features(:, 2) == 1);
    md_count = sum(user_features(:, 2) == 2);
    
    fprintf('  %s: %d samples (%d FD, %d MD) → %s\n', ...
            users{user}, num_user_samples, fd_count, md_count, filename);
end

fprintf('\nFEATURE EXTRACTION COMPLETED\n');
fprintf('All individual user files saved to %s/\n', output_dir);
fprintf('Feature matrix size: %d rows x %d columns\n', size(all_features, 1), size(all_features, 2));
fprintf('Total features per segment: %d (6 signals × 20 features each)\n', 6 * 20);