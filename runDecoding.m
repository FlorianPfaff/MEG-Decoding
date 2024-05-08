%% Tool to test different decoding methods
% Last author: Florian Pfaff, University of Stuttgart, pfaff@ias.uni-stuttgart.de
% Based on "runLasso.m" by Daniel Bush, UCL (2024) drdanielbush@gmail.com

function accuracies = runDecoding(dataFolder, parts, nFolds, windowSize, trainWindowCenter, ...
        nullWindowCenter, newFramerate, classifier, classifierParam, componentsPCA, frequencyRange)

    arguments
        dataFolder char = '.';
        parts (1, :) double {mustBeInteger, mustBePositive} = [1:4,6, 8:10, 13:27];
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
        classifier char = 'lasso';
        % Param of L1 regularisation of Lasso GLM or box constraint of SVM,
        % number of trees for RF, number of boosting iterations for GBM,
        % of neighbors for KNN, etc.
        classifierParam (1, 1) double = nan;
        % Number of components to retain after PCA. Set to inf to keep all components.
        componentsPCA (1, 1) double = inf;
        frequencyRange (1, 2) double = [0, inf];
    end

    trainWindow = trainWindowCenter + [-windowSize / 2, windowSize / 2];
    nullTimeWindow = nullWindowCenter + [-windowSize / 2, windowSize / 2];
    assert(any(isnan(nullTimeWindow)) || nullTimeWindow(end) <= trainWindow(1), 'Null window must be before train window')

    if any(isnan(classifierParam))
        classifierParam = getDefaultClassifierParam(classifier);
    end

    %% Assign some memory
    accuracies = nan(length(parts), 1);
    nFeatures = cell(length(parts), 1);

    %% Loop through each participant
    for p = 1:length(parts)

        % Load the data for that participant
        load([dataFolder filesep 'Part' int2str(parts(p)) 'Data.mat'], 'data');

        % Filter the data
        data = filterData(data, frequencyRange(1), frequencyRange(2));
        % Downsample the data, if required
        if newFramerate ~= inf
            data = downsampleData(data, newFramerate);
        end

        % Extract some basic information
        labels = data.trialinfo;
        nTrials = length(labels);
        nStim = length(unique(labels));

        % Create folds for cross-validation
        [allFeatures, labels, fold] = partitionData(data, nFolds, trainWindow, nullTimeWindow);
        nTrialsPerFold = nTrials / nFolds;
        % Assign some memory
        if componentsPCA == inf
            nFeats = size(allFeatures, 1);
        else
            nFeats = componentsPCA;
        end

        allFeats = nan(nFolds, nStim);
        predLbl = nan(nTrials, 1);

        % Loop through folds
        for f = 1:nFolds

            % Assign some memory
            allBeta = nan(nFeats, nStim);
            allInts = nan(nStim, 1);
            allPred = nan(nTrialsPerFold, nStim);

            % Get training and testing data
            trainFeatures = allFeatures(:, fold ~= f)';
            trainLabels = labels(fold ~= f)';
            testFeatures = allFeatures(:, fold == f & labels > 0)';

            if componentsPCA ~= inf
                % Apply PCA to the training data
                [reducedTrainFeatures, coeff, explainedVariance] = reduceFeaturesPCA(trainFeatures, componentsPCA);
                fprintf('Explained Variance by %d components: %.2f%%\n', componentsPCA, explainedVariance);

                % Transform the test features using the same PCA transformation
                testFeatures = testFeatures * coeff(:, 1:componentsPCA);

                % Update the trainFeatures and testFeatures variables to be used in classification
                trainFeatures = reducedTrainFeatures;
            end

            % Loop through stimuli, train and test the classifiers
            switch classifier
                case 'mostFrequentDummy'
                    predLbl(fold == f & labels > 0) = dummyClassifier(trainLabels);
                case 'always1Dummy'
                    predLbl(fold == f & labels > 0) = 1;
                case 'random-forest'
                    predLbl(fold == f & labels > 0) = trainAndPredictRandomForest(trainFeatures, trainLabels, testFeatures, classifierParam);
                case 'gradient-boosting'

                    for stim = 1:nStim
                        allPred(:, stim) = trainAndPredictGradientBoosting(trainFeatures, double(trainLabels == stim), testFeatures, classifierParam);
                    end

                    [~, predLbl(fold == f & labels > 0)] = max(allPred, [], 2);
                case 'multiclass-svm'
                    predLbl(fold == f & labels > 0) = trainAndPredictMulticlassSVM(trainFeatures, trainLabels, testFeatures, classifierParam, false);
                case 'multiclass-svm-weighted'
                    predLbl(fold == f & labels > 0) = trainAndPredictMulticlassSVM(trainFeatures, trainLabels, testFeatures, classifierParam, true);
                case 'knn'
                    predLbl(fold == f & labels > 0) = trainAndPredictKNN(trainFeatures, trainLabels, testFeatures, classifierParam);
                case 'lasso'

                    for stim = 1:nStim
                        [allPred(:, stim), allBeta(:, stim), allInts(stim)] = trainAndPredictForStimulusLassoGLM(trainFeatures, double(trainLabels == stim), testFeatures, classifierParam);
                    end

                    allFeats(f, :) = sum(allBeta ~= 0);
                    [~, predLbl(fold == f & labels > 0)] = max(allPred, [], 2);
                case 'svm-binary'

                    for stim = 1:nStim
                        allPred(:, stim) = trainAndPredictBinarySVM(trainFeatures, double(trainLabels == stim), testFeatures, classifierParam);
                    end

                    [~, predLbl(fold(labels > 0) == f)] = max(allPred, [], 2);
                otherwise
                    error('Invalid classifier type')
            end

        end

        predLbl'

        if all(predLbl == 0)
            fprintf(['All predictions are the null-class. Replace them to be fair with the binary classifiers for ' ...
                     'which one always decides on a label unequal to 0. Using 1.\n']);
            predLbl(predLbl == 0) = 1;
        elseif any(predLbl == 0)
            fprintf(['Some predictions are the null-class. Replace them to be fair with the binary classifiers for ' ...
                     'which one always decides on a label unequal to 0. Using least frequent label. \n']);

            % Find unique labels (excluding 0) and their occurrences
            nonzeroLabels = 1:nStim;
            occurrences = histcounts(predLbl(predLbl > 0), nStim);

            % Find the label with the minimum occurrence
            [~, minIndex] = min(occurrences);
            minLabel = nonzeroLabels(minIndex);

            % Assign this least frequent label to elements of predLbl that are 0
            predLbl(predLbl == 0) = minLabel;
        end

        % Compute overall accuracy
        accuracies(p) = mean(labels(labels > 0)' == predLbl);
        nFeatures{p} = allFeats;
        fprintf('Participant %d: %0.2f%% accuracy\n', parts(p), accuracies(p) * 100)

        if ~isnan(nFeatures{p})
            fprintf('Mean number of features: %0.2f\n', mean(nFeatures{p}, 'all'))
        end

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

function [trainData, labels, fold] = partitionData(data, nFolds, trainWindow, nullTimeWindow)

    arguments
        data struct
        nFolds (1, 1) double {mustBeInteger, mustBePositive}
        trainWindow (1, 2) double
        nullTimeWindow (1, 2) double
    end

    % Partition the data into blocks, identify training time and null time
    assert(any(isnan(nullTimeWindow)) || diff(nullTimeWindow) == diff(trainWindow), 'Train and null window must have the same length')
    assert(diff(trainWindow) >= 0, 'Train window ill defined')

    fold = ceil((1:length(data.trial)) / (length(data.trial) / nFolds));
    [~, trainBeginIndex] = min(abs(data.time{1} - trainWindow(1)));
    [~, trainEndIndex] = min(abs(data.time{1} - trainWindow(2)));
    trainDataCell = cellfun(@(x) reshape(x(:, trainBeginIndex:trainEndIndex), [], 1), data.trial, 'UniformOutput', false);

    labels = data.trialinfo;

    if any(isnan(nullTimeWindow))
        nullDataCell = {};
    elseif diff(nullTimeWindow) >= 0
        [~, nullBeginIndex] = min(abs(data.time{1} - nullTimeWindow(1)));
        nullEndIndex = nullBeginIndex + (trainEndIndex - trainBeginIndex);
        fold = [fold, fold];
        labels = [labels, zeros(1, length(data.trial))];
        nullDataCell = cellfun(@(x) reshape(x(:, nullBeginIndex:nullEndIndex), [], 1), data.trial, 'UniformOutput', false);
    else
        error('Invalid null window')
    end

    trainData = horzcat(trainDataCell{:}, nullDataCell{:});
end

function predictions = dummyClassifier(labels)
    % Dummy classifier that predicts the most frequent class
    frequencies = histcounts(labels(labels > 0));
    [~, mostFreqLabel] = max(frequencies);
    predictions = mostFreqLabel;
end

function [predictions, beta, intercept] = trainAndPredictForStimulusLassoGLM(trainFeatures, trainLabels, testFeatures, lambda)
    % Train a Lasso GLM model for binary classification
    [beta, fitInfo] = lassoglm(trainFeatures, trainLabels, 'binomial', 'Alpha', 1, 'Lambda', lambda, 'Standardize', true);
    intercept = fitInfo.Intercept; % Extract the intercept from fitInfo
    predictions = glmval([intercept; beta], testFeatures, 'logit');
end

function predictions = trainAndPredictBinarySVM(trainFeatures, trainLabels, testFeatures, boxConstraint)
    SVMModel = fitcsvm(trainFeatures, trainLabels, 'Standardize', true, 'KernelFunction', 'linear', 'BoxConstraint', boxConstraint);
    [~, score] = predict(SVMModel, testFeatures);
    predictions = score(:, 2); % The second column contains scores for the positive class
end

function predictions = trainAndPredictMulticlassSVM(trainFeatures, trainLabels, testFeatures, boxConstraint, weighted)
    t = templateSVM('Standardize', true, 'KernelFunction', 'linear', ...
        'BoxConstraint', boxConstraint);

    if ~weighted
        SVMModel = fitcecoc(trainFeatures, trainLabels, 'Coding', 'onevsone', 'Learners', t);
    else
        % Train a SVM model for multiclass classification using fitcecoc with weighting
        SVMModel = fitcecoc(trainFeatures, trainLabels, 'Coding', 'onevsone', ...
            'Learners', t, 'Weights', calculateObservationWeights(trainLabels));
    end

    % Make predictions for the current fold
    predictions = predict(SVMModel, testFeatures);

    function weightsObs = calculateObservationWeights(labels)
        % Calculate observation weights based on class frequency
        [frequencies, labelNames] = histcounts(categorical(labels));
        weightsClass = 1 ./ frequencies;
        weightsObs = nan(size(labels));

        for i = 1:numel(frequencies)
            weightsObs(labels == str2double(labelNames{i})) = weightsClass(i);
        end

        weightsObs = weightsObs / sum(weightsObs); % Normalize weights
    end

end

function predictions = trainAndPredictRandomForest(trainFeatures, trainLabels, testFeatures, classifierParam)
    % Train the Random Forest model
    RFModel = TreeBagger(classifierParam, trainFeatures, trainLabels, 'Method', 'classification', 'OOBPrediction', 'On', 'MinLeafSize', 5, 'OOBPredictorImportance', 'On');

    % Predict the responses for the testing set
    [predictions, ~] = predict(RFModel, testFeatures);
    % Convert predictions from cell to numeric, as predict returns cell array for classification
    predictions = str2double(predictions);
end

function predictions = trainAndPredictGradientBoosting(trainFeatures, trainLabels, testFeatures, classifierParam)
    % Train the Gradient Boosting model
    t = templateTree('MaxNumSplits', 20); % Customize decision tree template
    GBMModel = fitcensemble(trainFeatures, trainLabels, 'Method', 'LogitBoost', 'NumLearningCycles', classifierParam, 'Learners', t, 'LearnRate', 0.1);

    % Predict the responses for the testing set
    predictions = predict(GBMModel, testFeatures);
end

function predictions = trainAndPredictKNN(trainFeatures, trainLabels, testFeatures, numNeighbors)
    % Train and predict using a K-Nearest Neighbors (KNN) classifier
    %
    % Inputs:
    %   trainFeatures - A matrix where each row is a feature vector of a training example.
    %   trainLabels   - A column vector of labels for the training examples.
    %   testFeatures  - A matrix where each row is a feature vector of a test example to predict.
    %   numNeighbors  - The number of neighbors to use in the KNN classifier.
    %
    % Outputs:
    %   predictions   - A column vector of predicted labels for the test examples.

    % Create a KNN model using the training data
    KNNModel = fitcknn(trainFeatures, trainLabels, 'NumNeighbors', numNeighbors);

    % Predict the labels of the test data using the KNN model
    predictions = predict(KNNModel, testFeatures);
end

function [reducedFeatures, coeff, explainedVariance] = reduceFeaturesPCA(features, nComponents)
    % reduceFeaturesPCA Perform dimensionality reduction on feature set using PCA
    %
    % Inputs:
    %   features - The original set of features (observations x features)
    %   nComponents - The number of principal components to retain
    %
    % Outputs:
    %   reducedFeatures - The reduced set of features after PCA (observations x nComponents)
    %   explainedVariance - The percentage of variance explained by the selected components

    % Ensure nComponents does not exceed the number of features
    nComponents = min(nComponents, size(features, 2));

    % Perform PCA on the features
    [coeff, ~, ~, ~, explained] = pca(features);

    % Select the first nComponents
    reducedCoeff = coeff(:, 1:nComponents);
    reducedFeatures = features * reducedCoeff;

    % Calculate the explained variance of the selected components
    explainedVariance = sum(explained(1:nComponents));

end

function newData = filterData(data, lowFreq, highFreq)
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
