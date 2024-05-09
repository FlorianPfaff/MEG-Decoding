%% Tool to test different decoding methods
% Last author: Florian Pfaff, University of Stuttgart, pfaff@ias.uni-stuttgart.de
% Based on "runLasso.m" by Daniel Bush, UCL (2024) drdanielbush@gmail.com

function accuracies = runDecoding(dataFolder, parts, nFolds, windowSize, trainWindowCenter, ...
        nullWindowCenter, newFramerate, classifier, classifierParam, componentsPCA, frequencyRange)

    arguments
        dataFolder char = '.';
        parts (1, :) double {mustBeInteger, mustBePositive} = [2]
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
        % Extract some basic information
        labels = data.trialinfo;
        nTrials = length(labels);
        nStim = length(unique(labels));
        [stimuliFeaturesCell, nullFeaturesCell] = preprocessFeatures(data, frequencyRange, newFramerate, windowSize, trainWindowCenter, nullWindowCenter);
        allFeatures = horzcat(stimuliFeaturesCell{:}, nullFeaturesCell{:});

        % Create folds
        fold = ceil((1:length(data.trial)) / (length(data.trial) / nFolds));
        if ~isempty(nullFeaturesCell)
            % Assing null data to the same fold as the corresponding stimulus data
            fold = [fold, fold];
            labels = [labels, zeros(1, length(data.trial))];
        end

        nTrialsPerFold = nTrials / nFolds;

        allFeats = nan(nFolds, nStim);
        predLbl = nan(nTrials, 1);

        % Loop through folds
        for f = 1:nFolds

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

            if contains(classifier, {'gradient-boosting', 'lasso', 'svm-binary'})
                % Iterate over all stimuli for binary classifiers
                models = cell(1, nStim);
                for stim = 1:nStim
                    switch classifier
                        case 'gradient-boosting'
                            models{stim} = trainGradientBoosting(trainFeatures, double(trainLabels == stim), testFeatures, classifierParam);
                            allPred(:, stim) = generatePredictionsFromModel(testFeatures, models{stim});
                        case 'lasso'
                            models{stim} = trainForStimulusLassoGLM(trainFeatures, double(trainLabels == stim), classifierParam);
                            allPred(:, stim) = glmval([models{stim}.intercept; models{stim}.beta], testFeatures, 'logit');
                        case 'svm-binary'
                            models{stim} = trainBinarySVM(trainFeatures, double(trainLabels == stim), testFeatures, classifierParam);
                            [~, score] = predict(models{stim}, testFeatures);
                            allPred(:, stim) = score(:, 2); % The second column contains scores for the positive class
                        otherwise
                            error('Unsupported classifier.')
                    end
                end
                % Set to most suitable class
                [~, predLbl(fold == f & labels > 0)] = max(allPred, [], 2);
            else
                % No iteration for multiclass classifiers
                % Train
                model = trainMulticlassClassifier(trainFeatures, trainLabels, classifier, classifierParam);
                % Predict
                predLbl(fold == f & labels > 0) = generatePredictionsFromModel(testFeatures, model, classifier);
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

function model = trainGradientBoosting(trainFeatures, trainLabels, classifierParam)
    % Train the Gradient Boosting model
    t = templateTree('MaxNumSplits', 20); % Customize decision tree template
    model = fitcensemble(trainFeatures, trainLabels, 'Method', 'LogitBoost', 'NumLearningCycles', classifierParam, 'Learners', t, 'LearnRate', 0.1);
end

function modelParams = trainForStimulusLassoGLM(trainFeatures, trainLabels, lambda)
    % Train a Lasso GLM model for binary classification
    [beta, fitInfo] = lassoglm(trainFeatures, trainLabels, 'binomial', 'Alpha', 1, 'Lambda', lambda, 'Standardize', true);
    intercept = fitInfo.Intercept; % Extract the intercept from fitInfo
    modelParams = struct('beta', beta, 'intercept', intercept);
end

function model = trainBinarySVM(trainFeatures, trainLabels, boxConstraint)
    model = fitcsvm(trainFeatures, trainLabels, 'Standardize', true, 'KernelFunction', 'linear', 'BoxConstraint', boxConstraint);
end
