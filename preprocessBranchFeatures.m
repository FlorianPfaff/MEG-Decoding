function [stimuliFeaturesByBranch, nullFeaturesByBranch, branchInfo] = preprocessBranchFeatures(data, frequencyRange, newFramerate, windowSize, trainWindowCenter, nullWindowCenter, branchInfo)
    % preprocessBranchFeatures Build parallel temporal/spatial feature branches.
    %
    % This is the feature extractor for the 'branch-fusion-svm' classifier.
    % It keeps the source-only hygiene of the existing pipeline: all branch
    % definitions are fixed from time/channel metadata, while all fitted
    % transforms such as PCA and SVM standardization remain inside the
    % training fold.
    %
    % Outputs are cell arrays with one entry per branch. Each branch entry is
    % itself a cell array with one column-vector feature per trial, matching
    % the convention used by preprocessFeatures.

    if nargin < 7
        branchInfo = [];
    end

    % Filter the data
    data = filterFeaturesForBranches(data, frequencyRange(1), frequencyRange(2));

    % Downsample the data, if required
    if newFramerate ~= inf
        data = downsampleDataForBranches(data, newFramerate);
    end

    if isempty(branchInfo)
        temporalWindows = getDefaultTemporalWindows(data, windowSize, trainWindowCenter);
        spatialGroups = inferSpatialGroups(data);
        branchInfo = makeBranchInfo(temporalWindows, spatialGroups);
    end

    nBranches = numel(branchInfo);
    stimuliFeaturesByBranch = cell(1, nBranches);
    nullFeaturesByBranch = cell(1, nBranches);

    for b = 1:nBranches
        timeIndices = getTimeIndices(data.time{1}, branchInfo(b).timeWindow);
        channelIndices = branchInfo(b).channels;

        stimuliFeaturesByBranch{b} = cellfun(@(x) reshape(x(channelIndices, timeIndices), [], 1), ...
            data.trial, 'UniformOutput', false);

        if any(isnan(nullWindowCenter))
            nullFeaturesByBranch{b} = {};
        else
            duration = diff(branchInfo(b).timeWindow);
            nullTimeWindow = nullWindowCenter + [-duration / 2, duration / 2];
            assert(nullTimeWindow(end) <= branchInfo(b).timeWindow(1), 'Null window must be before train window')
            assert(nullTimeWindow(end) <= 0, 'Null window should not contain positive time points')

            nullBeginIndex = getNearestTimeIndex(data.time{1}, nullTimeWindow(1));
            nullEndIndex = nullBeginIndex + numel(timeIndices) - 1;
            assert(nullEndIndex <= numel(data.time{1}), 'Null window exceeds available time range')
            nullTimeIndices = nullBeginIndex:nullEndIndex;

            nullFeaturesByBranch{b} = cellfun(@(x) reshape(x(channelIndices, nullTimeIndices), [], 1), ...
                data.trial, 'UniformOutput', false);
        end
    end
end

function temporalWindows = getDefaultTemporalWindows(data, windowSize, trainWindowCenter)
    mainWindow = trainWindowCenter + [-windowSize / 2, windowSize / 2];

    candidateNames = {'main', 'early_visual', 'mid_visual', 'late_visual'};
    candidateWindows = [
        mainWindow;
        0.05, 0.15;
        0.15, 0.30;
        0.30, 0.50
    ];

    temporalWindows = struct('name', {}, 'window', {});
    seenKeys = {};
    for i = 1:size(candidateWindows, 1)
        currWindow = candidateWindows(i, :);
        if i ~= 1 && ~isWindowInsideData(currWindow, data.time{1})
            continue
        end

        key = sprintf('%.6f_%.6f', currWindow(1), currWindow(2));
        if ismember(key, seenKeys)
            continue
        end

        temporalWindows(end + 1).name = candidateNames{i}; %#ok<AGROW>
        temporalWindows(end).window = currWindow;
        seenKeys{end + 1} = key; %#ok<AGROW>
    end

    assert(~isempty(temporalWindows), 'No valid temporal branch windows found')
end

function isInside = isWindowInsideData(timeWindow, time)
    tolerance = 0.5 * median(diff(time));
    isInside = timeWindow(1) >= min(time) - tolerance && ...
        timeWindow(2) <= max(time) + tolerance && ...
        timeWindow(2) > timeWindow(1);
end

function spatialGroups = inferSpatialGroups(data)
    nChannels = size(data.trial{1}, 1);
    spatialGroups = struct('name', 'all', 'indices', 1:nChannels);

    channelPositions = getChannelPositions(data, nChannels);
    if isempty(channelPositions) || size(channelPositions, 2) < 2
        return
    end

    x = channelPositions(:, 1);
    y = channelPositions(:, 2);
    if any(~isfinite(x)) || any(~isfinite(y))
        return
    end

    spatialGroups = addSpatialGroup(spatialGroups, 'posterior', y <= median(y));
    spatialGroups = addSpatialGroup(spatialGroups, 'left', x < median(x));
    spatialGroups = addSpatialGroup(spatialGroups, 'right', x >= median(x));
end

function spatialGroups = addSpatialGroup(spatialGroups, name, mask)
    mask = mask(:)';
    indices = find(mask);
    nChannels = numel(mask);

    if numel(indices) < 2 || numel(indices) == nChannels
        return
    end

    for i = 1:numel(spatialGroups)
        if isequal(spatialGroups(i).indices, indices)
            return
        end
    end

    spatialGroups(end + 1).name = name; %#ok<AGROW>
    spatialGroups(end).indices = indices;
end

function channelPositions = getChannelPositions(data, nChannels)
    channelPositions = [];
    labels = getDataLabels(data, nChannels);

    if isfield(data, 'grad') && isfield(data.grad, 'chanpos')
        candidatePositions = data.grad.chanpos;
        candidateLabels = getFieldIfPresent(data.grad, 'label');
        channelPositions = alignChannelPositions(labels, candidateLabels, candidatePositions, nChannels);
        if ~isempty(channelPositions)
            return
        end

        if size(candidatePositions, 1) >= nChannels
            channelPositions = candidatePositions(1:nChannels, :);
            return
        end
    end

    if isfield(data, 'elec') && isfield(data.elec, 'chanpos')
        candidatePositions = data.elec.chanpos;
        candidateLabels = getFieldIfPresent(data.elec, 'label');
        channelPositions = alignChannelPositions(labels, candidateLabels, candidatePositions, nChannels);
        if ~isempty(channelPositions)
            return
        end

        if size(candidatePositions, 1) >= nChannels
            channelPositions = candidatePositions(1:nChannels, :);
            return
        end
    end
end

function labels = getDataLabels(data, nChannels)
    labels = {};
    if isfield(data, 'label')
        labels = data.label;
    elseif isfield(data, 'grad') && isfield(data.grad, 'label') && numel(data.grad.label) >= nChannels
        labels = data.grad.label(1:nChannels);
    end

    if isstring(labels)
        labels = cellstr(labels);
    end
    labels = labels(:);
end

function value = getFieldIfPresent(s, fieldName)
    if isfield(s, fieldName)
        value = s.(fieldName);
    else
        value = {};
    end
end

function channelPositions = alignChannelPositions(labels, candidateLabels, candidatePositions, nChannels)
    channelPositions = [];
    if isempty(labels) || isempty(candidateLabels) || size(candidatePositions, 1) < nChannels
        return
    end

    if isstring(candidateLabels)
        candidateLabels = cellstr(candidateLabels);
    end
    candidateLabels = candidateLabels(:);

    if numel(labels) ~= nChannels
        return
    end

    alignedPositions = nan(nChannels, size(candidatePositions, 2));
    for i = 1:nChannels
        matchIndex = find(strcmp(candidateLabels, labels{i}), 1);
        if isempty(matchIndex)
            return
        end
        alignedPositions(i, :) = candidatePositions(matchIndex, :);
    end
    channelPositions = alignedPositions;
end

function branchInfo = makeBranchInfo(temporalWindows, spatialGroups)
    branchInfo = struct('name', {}, 'timeWindow', {}, 'channels', {}, 'temporalBranch', {}, 'channelGroup', {});
    branchIdx = 0;

    for t = 1:numel(temporalWindows)
        for g = 1:numel(spatialGroups)
            branchIdx = branchIdx + 1;
            branchInfo(branchIdx).name = [temporalWindows(t).name '_' spatialGroups(g).name];
            branchInfo(branchIdx).timeWindow = temporalWindows(t).window;
            branchInfo(branchIdx).channels = spatialGroups(g).indices;
            branchInfo(branchIdx).temporalBranch = temporalWindows(t).name;
            branchInfo(branchIdx).channelGroup = spatialGroups(g).name;
        end
    end
end

function indices = getTimeIndices(time, timeWindow)
    assert(diff(timeWindow) >= 0, 'Time window ill defined')
    beginIndex = getNearestTimeIndex(time, timeWindow(1));
    endIndex = getNearestTimeIndex(time, timeWindow(2));
    if endIndex < beginIndex
        tmp = beginIndex;
        beginIndex = endIndex;
        endIndex = tmp;
    end
    indices = beginIndex:endIndex;
end

function index = getNearestTimeIndex(time, targetTime)
    [~, index] = min(abs(time - targetTime));
end

function newData = filterFeaturesForBranches(data, lowFreq, highFreq)
    newData = data;

    if isempty(data.time{1})
        error('Time vector is empty or not provided correctly.');
    end

    Fs = 1 / mean(diff(data.time{1}));
    assert(lowFreq >= 0, 'Low frequency must be greater than or equal to 0')
    assert(highFreq >= 0, 'High frequency must be greater than or equal to 0')
    assert(highFreq >= lowFreq, 'High frequency must be greater than or equal to low frequency')

    if lowFreq == 0 && highFreq == inf
        return
    elseif lowFreq == 0 && highFreq ~= inf
        [b, a] = butter(4, highFreq / (Fs / 2), 'low');
    elseif lowFreq ~= 0 && highFreq ~= inf
        [b, a] = butter(4, [lowFreq highFreq] / (Fs / 2), 'bandpass');
    else
        error('Highpass filter not supported.')
    end

    for i = 1:length(data.trial)
        newData.trial{i} = filtfilt(b, a, data.trial{i}.');
        newData.trial{i} = newData.trial{i}.';
    end
end

function data = downsampleDataForBranches(data, newFramerate)
    rawFs = round(1 / median(diff(data.time{1})));

    if ~isempty(newFramerate) && rawFs ~= newFramerate
        newT = data.time{1}(1):1 / newFramerate:data.time{1}(end);

        for t = 1:length(data.trial)
            data.trial{t} = detrend(data.trial{t}')';
            data.trial{t} = interp1(data.time{t}, data.trial{t}', newT)';
            data.time{t} = newT;
        end
    end
end
