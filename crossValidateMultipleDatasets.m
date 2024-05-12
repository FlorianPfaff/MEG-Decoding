function accuracies = crossValidateMultipleDatasets(dataFolder, participantIDs, nFolds, windowSize, trainWindowCenter, ...
    nullWindowCenter, newFramerate, classifier, classifierParam, componentsPCA, frequencyRange)
    % Simple wrapper for crossValidateSingleDataset to use it with multiple datasets
    arguments
        dataFolder char = '.';
        participantIDs (1, 1) double {mustBeInteger, mustBePositive} = 2
        nFolds (1, 1) double {mustBeInteger, mustBePositive} = 10;
        % Window size in seconds
        windowSize (1, 1) double = 0.1;
        % All times are relative to the onset of the stimulus
        % If a single number is given, only one point in time is used.
        % Use two numbers to specify a time window, e.g., [0.2, 0.3].
        % Original was 0.2
        trainWindowCenter (1, 1) double = 0.2;
        % Set to nan for disabling extraction of null data. Default: -0.2
        nullWindowCenter (1, 1) double = -0.2;
        % Set newFramerate to inf to disable downsampling. Set to '100' to emulate the behavior of runLasso.m
        newFramerate (1, 1) double = inf;
        % Type of classifier to use, e.g. 'lasso', 'multiclass-svm', 'random-forest', 'gradient-boosting', 'knn', 'mostFrequentDummy', 'always1Dummy'.
        classifier char = 'multiclass-svm';
        % Param of L1 regularisation of Lasso GLM or box constraint of SVM,
        % number of trees for RF, number of boosting iterations for GBM,
        % of neighbors for KNN, etc.
        classifierParam (1, 1) double = nan;
        % Number of components to retain after PCA. Set to inf to keep all components.
        componentsPCA (1, 1) double = 100;
        frequencyRange (1, 2) double = [0, inf];
    end

    accuracies = nan(1, length(participantIDs));

    for i = 1:length(participantIDs)
        accuracies(i) = crossValidateSingleDataset(dataFolder, participantIDs(i), nFolds, windowSize, trainWindowCenter, ...
            nullWindowCenter, newFramerate, classifier, classifierParam, componentsPCA, frequencyRange);
    end
    sprintf('Mean accuracy: %f', mean(accuracies))
end