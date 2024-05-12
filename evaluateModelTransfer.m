function accuracy = evaluateModelTransfer(dataFolder, parts, windowSize, trainWindowCenter, ...
    nullWindowCenter, newFramerate, classifier, classifierParam, componentsPCA, frequencyRange)

    arguments
        dataFolder char = '.';
        parts (1, 1) double {mustBeInteger, mustBePositive} = [2]
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

    if any(isnan(classifierParam))
        classifierParam = getDefaultClassifierParam(classifier);
    end
    trainExpData = load([dataFolder filesep 'Part' int2str(parts) 'Data.mat']);
    valExpData = load([dataFolder filesep 'Part' int2str(parts) 'CueData.mat']);

    labelsTrainExp = trainExpData.data.trialinfo;
    labelsValExp = valExpData.data.trialinfo;
    assert(diff(valExpData.data.time{1}(1:2))-diff(valExpData.data.time{1}(1:2))==0, 'Sampling rate of the two experiments must match.')
    if ~isempty(setdiff(labelsTrainExp, labelsValExp))
        warning('There are labels in the training or validation experiment that are not in the other experiment.')
    end
    [stimuliFeaturesCellTrainExp, nullFeaturesCellTrainExp] = preprocessFeatures(trainExpData.data, frequencyRange, newFramerate, windowSize, trainWindowCenter, nullWindowCenter);
    [stimuliFeaturesCellValExp, ~] = preprocessFeatures(valExpData.data, frequencyRange, newFramerate, windowSize, trainWindowCenter, nan);
    featuresTrainExp = horzcat(stimuliFeaturesCellTrainExp{:}, nullFeaturesCellTrainExp{:})';
    labelsTrainExp = [labelsTrainExp, zeros(1, length(nullFeaturesCellTrainExp))];
    featuresValExp = horzcat(stimuliFeaturesCellValExp{:})';
    
    if componentsPCA ~= inf
        [featuresTrainExp, coeff, featuresTrainExpMean, explainedVariance] = reduceFeaturesPCA(featuresTrainExp, componentsPCA);
        fprintf('Explained Variance by %d components: %.2f%%\n', componentsPCA, explainedVariance);
        % Transform the features for the validation experiment using the same PCA transformation
        featuresValExp = (featuresValExp - featuresTrainExpMean) * coeff(:, 1:componentsPCA);
    end

    model = trainMulticlassClassifier(featuresTrainExp, labelsTrainExp, classifier, classifierParam);
    predictionsValExp = generatePredictionsFromModel(featuresValExp, model, classifier);


    accuracy = mean(predictionsValExp == labelsValExp')
end