function exportTrialTensorForPyTorch(dataFolder, participantIDs, outputFile, windowSize, trainWindowCenter, nullWindowCenter, newFramerate, frequencyRange)
% exportTrialTensorForPyTorch Export FieldTrip MEG trials for PyTorch LOSO baselines.
%
% The output MAT file contains:
%   X        : trials x channels x time tensor
%   y        : trials x 1 labels
%   subjects : trials x 1 participant IDs
%
% This uses the same preprocessFeatures.m path as the existing MATLAB baselines
% for filtering, downsampling, and window extraction. Null-window trials are not
% exported because the PyTorch baselines are multiclass stimulus classifiers.

    arguments
        dataFolder char = '.';
        participantIDs (1, :) double {mustBeInteger, mustBePositive} = [1:4, 6, 8:10, 13:27];
        outputFile char = 'meg_trials_for_pytorch.mat';
        windowSize (1, 1) double = 0.2;
        trainWindowCenter (1, 1) double = 0.2;
        nullWindowCenter (1, 1) double = nan; %#ok<INUSA> Kept for API symmetry; neural export omits null trials.
        newFramerate (1, 1) double = inf;
        frequencyRange (1, 2) double = [0, inf];
    end

    XParts = {};
    yParts = {};
    subjectParts = {};

    for i = 1:numel(participantIDs)
        participantID = participantIDs(i);
        loaded = load([dataFolder filesep 'Part' int2str(participantID) 'Data.mat'], 'data');
        data = loaded.data;
        labels = data.trialinfo(:);
        nChannels = size(data.trial{1}, 1);

        % Disable null-window extraction for the neural baselines.
        [stimuliFeaturesCell, ~] = preprocessFeatures(data, frequencyRange, newFramerate, windowSize, trainWindowCenter, nan);
        nTrials = numel(stimuliFeaturesCell);
        nWindowSamples = numel(stimuliFeaturesCell{1}) / nChannels;
        assert(mod(nWindowSamples, 1) == 0, 'Window feature length is not divisible by number of channels.');

        XPart = zeros(nTrials, nChannels, nWindowSamples, 'single');
        for t = 1:nTrials
            XPart(t, :, :) = single(reshape(stimuliFeaturesCell{t}, nChannels, nWindowSamples));
        end

        XParts{end + 1} = XPart; %#ok<AGROW>
        yParts{end + 1} = labels; %#ok<AGROW>
        subjectParts{end + 1} = participantID * ones(nTrials, 1); %#ok<AGROW>
    end

    X = cat(1, XParts{:}); %#ok<NASGU>
    y = cat(1, yParts{:}); %#ok<NASGU>
    subjects = cat(1, subjectParts{:}); %#ok<NASGU>
    exportedParticipantIDs = participantIDs; %#ok<NASGU>
    preprocessing = struct('windowSize', windowSize, 'trainWindowCenter', trainWindowCenter, ...
        'newFramerate', newFramerate, 'frequencyRange', frequencyRange); %#ok<NASGU>

    save(outputFile, 'X', 'y', 'subjects', 'exportedParticipantIDs', 'preprocessing', '-v7');
    fprintf('Exported %d trials to %s\n', size(X, 1), outputFile);
end
