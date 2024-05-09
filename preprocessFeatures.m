function [stimuliFeaturesCell, nullFeaturesCell] = preprocessFeatures(data, frequencyRange, newFramerate, windowSize, trainWindowCenter, nullWindowCenter)
    % Filter the data
    data = filterFeatures(data, frequencyRange(1), frequencyRange(2));
    % Downsample the data, if required
    if newFramerate ~= inf
        data = downsampleData(data, newFramerate);
    end
    trainWindow = trainWindowCenter + [-windowSize / 2, windowSize / 2];
    nullTimeWindow = nullWindowCenter + [-windowSize / 2, windowSize / 2];
    assert(any(isnan(nullTimeWindow)) || nullTimeWindow(end) <= trainWindow(1), 'Null window must be before train window')
    % Extract Windows
    [stimuliFeaturesCell, nullFeaturesCell] = extractWindows(data, trainWindow, nullTimeWindow);
end

function newData = filterFeatures(data, lowFreq, highFreq)
    newData = data; % Initialize newData with the input data structure

    % Determine the sampling rate from the time vector of the first trial
    % Assumes uniform sampling rate across all trials
    if ~isempty(data.time{1})
        Fs = 1 / mean(diff(data.time{1}));
    else
        error('Time vector is empty or not provided correctly.');
    end
    assert(lowFreq>=0, 'Low frequency must be greater than or equal to 0')
    assert(highFreq>=0, 'High frequency must be greater than or equal to 0')
    assert(highFreq>=lowFreq, 'High frequency must be greater than or equal to low frequency')
    % Design the filter based on the input frequencies
    if lowFreq == 0 && highFreq == inf
        % Nothing to do
        return
    elseif lowFreq == 0 && highFreq ~= inf
        % Lowpass filter design
        [b, a] = butter(4, highFreq / (Fs / 2), 'low');
    elseif lowFreq ~= 0 && highFreq ~= inf
        % Bandpass filter design
        [b, a] = butter(4, [lowFreq highFreq] / (Fs / 2), 'bandpass');
    else
        error("Highpass filter not supported.")
    end

    % Apply the designed filter to each trial
    for i = 1:length(data.trial)
        % Apply the filter
        % Transpose data.trial{i} to match filtfilt dimension requirements (time dimension first)
        newData.trial{i} = filtfilt(b, a, data.trial{i}.');
        % Transpose back to original dimensions (sensors, timesteps)
        newData.trial{i} = newData.trial{i}.';
    end
end

function data = downsampleData(data, newFramerate)
    % Downsample the MEG data to a new sampling frequency if required

    % Determine the original sampling frequency
    rawFs = round(1 / median(diff(data.time{1})));

    % Only proceed if newFramerate is different from rawFs
    if ~isempty(newFramerate) && rawFs ~= newFramerate
        % Define new time vector based on newFramerate
        newT = data.time{1}(1):1 / newFramerate:data.time{1}(end);

        % Loop through each trial to downsample
        for t = 1:length(data.trial)
            % Detrend and interpolate data for this trial
            data.trial{t} = detrend(data.trial{t}')'; % Detrend data before interpolation
            data.trial{t} = interp1(data.time{t}, data.trial{t}', newT)'; % Interpolate data
            data.time{t} = newT; % Update time vector for this trial
        end
    end
end

function [stimuliFeaturesCell, nullFeaturesCell] = extractWindows(data, trainWindow, nullTimeWindow)
    arguments
        data struct
        trainWindow (1, 2) double
        nullTimeWindow (1, 2) double
    end

    % Partition the data into blocks, identify training time and null time
    assert(any(isnan(nullTimeWindow)) || diff(nullTimeWindow) == diff(trainWindow), 'Train and null window must have the same length')
    assert(diff(trainWindow) >= 0, 'Train window ill defined')

    
    [~, trainBeginIndex] = min(abs(data.time{1} - trainWindow(1)));
    [~, trainEndIndex] = min(abs(data.time{1} - trainWindow(2)));
    stimuliFeaturesCell = cellfun(@(x) reshape(x(:, trainBeginIndex:trainEndIndex), [], 1), data.trial, 'UniformOutput', false);

    if any(isnan(nullTimeWindow))
        nullFeaturesCell = {};
    elseif diff(nullTimeWindow) >= 0
        [~, nullBeginIndex] = min(abs(data.time{1} - nullTimeWindow(1)));
        nullEndIndex = nullBeginIndex + (trainEndIndex - trainBeginIndex);
        nullFeaturesCell = cellfun(@(x) reshape(x(:, nullBeginIndex:nullEndIndex), [], 1), data.trial, 'UniformOutput', false);
    else
        error('Invalid null window')
    end
end