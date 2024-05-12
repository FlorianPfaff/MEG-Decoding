function paramTable = runGridSearchDecoding(dataFolder, participantIDs, nFolds, windowSizes, trainWindowCenters, ...
        nullWindowCenters, newFramerates, classifiers, classifierParams, componentsPCAs, frequencyRanges)

    arguments
        dataFolder char = '.';
        participantIDs (1, :) double {mustBeInteger, mustBePositive} = [1:4, 6, 8:10, 13:27];
        nFolds (1, 1) double {mustBeInteger, mustBePositive} = 10;
        % Window size in seconds
        windowSizes (1, :) double = [0.2];
        % All times are relative to the onset of the stimulus
        % If a single number is given, only one point in time is used.
        % Use two numbers to specify a time window, e.g., [0.2, 0.3].
        % Original was 0.2
        trainWindowCenters (1, :) double = 0.2;
        % Set to nan for disabling extraction of null data. Default: -0.2
        nullWindowCenters (1, :) double = nan;
        % Set newFramerates to inf to disable downsampling. Set to '100' to emulate the behavior of runLasso.m
        newFramerates (1, :) double = inf % [100, inf];
        % Type of classifier to use, e.g. 'lasso', 'multiclass-svm', 'random-forest', 'gradient-boosting', 'knn', 'mostFrequentDummy', 'always1Dummy'.
        classifiers cell = {'lasso', 'multiclass-svm-weighted'};
        % Param of L1 regularisation of Lasso GLM or box constraint of SVM,
        % number of trees for RF, number of boosting iterations for GBM,
        % of neighbors for KNN, etc.
        classifierParams (1, :) double = nan;
        % Number of components to retain after PCA. Set to inf to keep all components.
        componentsPCAs (1, :) double = [200];
        % Frequency range to use for the decoding. Set to [0, inf] to use all frequencies.
        % To try multiple, use a cell array, e.g. {[0, 100], [0, 30]}.
        frequencyRanges = {[0, inf];[0,100]};
    end

    saveFolder = dataFolder;
    if ~iscell(frequencyRanges)
        frequencyRanges = {frequencyRanges};
    end
    % Generate all combinations of the parameters
    paramCell = cell(1, 8);
    [paramCell{:}] = ndgrid(classifiers, classifierParams, componentsPCAs, newFramerates, trainWindowCenters, nullWindowCenters, windowSizes, frequencyRanges);
    paramCell_flat = cellfun(@(x) x(:), paramCell, 'UniformOutput', false);

    % Convert the cell array and vectors to a table
    paramTable = table(paramCell_flat{:}, ...
        'VariableNames', {'Classifier', 'ClassifierParam', 'ComponentsPCA', 'Framerate', ...
           'TrainWindowCenter', 'NullWindowCenter', 'WindowSize', 'FrequencyRange'});


    for i = 1:size(paramTable, 1)
        if isnan(paramTable.ClassifierParam(i))
            paramTable.ClassifierParam(i) = getDefaultClassifierParam(paramTable.Classifier{i});
        end
    end
    % Placeholder for the results
    numCombinations = size(paramTable, 1);
    accuracies = repmat({NaN(size(participantIDs))}, [numCombinations, 1]);
    meanAccuracies = NaN(numCombinations, 1);
    % Display the table (optional)
    disp(paramTable);

    % Parallel execution
    parfor i = 1:numCombinations
        currRow = paramTable(i, :);
        classifier = [currRow.Classifier{:}];
        componentsPCA = currRow.ComponentsPCA;
        windowSize = currRow.WindowSize;
        nullWindowCenter = currRow.NullWindowCenter;
        newFramerate = currRow.Framerate;
        classifierParam = currRow.ClassifierParam;
        frequencyRange = currRow.FrequencyRange{:};

        % Run the decoding function with the current set of parameters
        for j = 1:numel(participantIDs)
            % Evaluate them one by one to avoid losing results on failure on any individual case
            try
                accuracies{i}(j) = crossValidateMultipleDatasets(dataFolder, participantIDs(j), nFolds, windowSize, trainWindowCenters, ...
                    nullWindowCenter, newFramerate, classifier, classifierParam, componentsPCA, frequencyRange);
            catch ME
                fprintf('Error with combination %d: %s\n', i, ME.message);
            end
        end
        meanAccuracies(i) = mean(accuracies{i}, 'omitnan');
    end

    resultsTable = paramTable;
    resultsTable.TrainWindow = [resultsTable.TrainWindowCenter - 0.5 * resultsTable.WindowSize, resultsTable.TrainWindowCenter + 0.5 * resultsTable.WindowSize];
    resultsTable.NullWindow = [resultsTable.NullWindowCenter - 0.5 * resultsTable.WindowSize, resultsTable.NullWindowCenter + 0.5 * resultsTable.WindowSize];
    resultsTable.TrainWindowCenter = [];
    resultsTable.NullWindowCenter = [];
    resultsTable.WindowSize = [];
    resultsTable.Accuracies = cat(1, accuracies{:});
    resultsTable.FrequencyRange = cell2mat(resultsTable.FrequencyRange);
    resultsTable.MeanAccuracies = meanAccuracies;
    % Display the results
    disp(resultsTable);
    % Save the results
    dateAndTime = datetime;
    filename = fullfile(saveFolder, sprintf(['results', '-%4d-%02d-%02d--%02d-%02d-%02d.mat'], dateAndTime.Year, dateAndTime.Month, dateAndTime.Day, dateAndTime.Hour, dateAndTime.Minute, floor(dateAndTime.Second)));
    [~, hostname] = system('hostname');
    save(filename, 'resultsTable', 'hostname', '-v7.3');
end
