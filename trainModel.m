function model = trainModel(trainFeatures, trainLabels, classifier, classifierParam)
    switch classifier
        case 'mostFrequentDummy'
            model = trainDummyClassifier(trainLabels);
        case 'always1Dummy'
            model = @(input) 1 * ones(size(input, 2));
        case 'random-forest'
            model = trainRandomForest(trainFeatures, trainLabels, classifierParam);
        case 'gradient-boosting'
            model = trainGradientBoosting(trainFeatures, trainLabels, classifierParam);
        case 'multiclass-svm'
            model = trainMulticlassSVM(trainFeatures, trainLabels, classifierParam, false);
        case 'multiclass-svm-weighted'
            model = trainMulticlassSVM(trainFeatures, trainLabels, classifierParam, true);
        case 'knn'
            model = trainKNN(trainFeatures, trainLabels, classifierParam);
        case 'lasso'
            model = trainLassoGLM(trainFeatures, trainLabels, classifierParam);
        case 'svm-binary'
            model = trainBinarySVM(trainFeatures, trainLabels, classifierParam);
        otherwise
            error('Invalid classifier type');
    end
end

function model = trainDummyClassifier(labels)
    % Dummy classifier that predicts the most frequent class
    frequencies = histcounts(labels(labels > 0));
    [~, mostFreqLabel] = max(frequencies);
    
    model = @(input) mostFreqLabel * ones(size(input, 2));
end

function model = trainLassoGLM(trainFeatures, trainLabels, lambda)
    % Train a Lasso GLM model for binary classification
    [beta, fitInfo] = lassoglm(trainFeatures, trainLabels, 'binomial', 'Alpha', 1, 'Lambda', lambda, 'Standardize', true);
    intercept = fitInfo.Intercept; % Extract the intercept from fitInfo
    
    model = struct('beta', beta, 'intercept', intercept);
end

function model = trainBinarySVM(trainFeatures, trainLabels, boxConstraint)
    % Train a binary SVM model
    model = fitcsvm(trainFeatures, trainLabels, 'Standardize', true, 'KernelFunction', 'linear', 'BoxConstraint', boxConstraint);
end

function model = trainMulticlassSVM(trainFeatures, trainLabels, boxConstraint, weighted)
    t = templateSVM('Standardize', true, 'KernelFunction', 'linear', 'BoxConstraint', boxConstraint);
    
    if ~weighted
        model = fitcecoc(trainFeatures, trainLabels, 'Coding', 'onevsone', 'Learners', t);
    else
        model = fitcecoc(trainFeatures, trainLabels, 'Coding', 'onevsone', 'Learners', t, 'Weights', calculateObservationWeights(trainLabels));
    end

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

function model = trainRandomForest(trainFeatures, trainLabels, classifierParam)
    % Train the Random Forest model
    model = TreeBagger(classifierParam, trainFeatures, trainLabels, 'Method', 'classification', 'OOBPrediction', 'On', 'MinLeafSize', 5, 'OOBPredictorImportance', 'On');
end

function model = trainGradientBoosting(trainFeatures, trainLabels, classifierParam)
    % Train the Gradient Boosting model
    t = templateTree('MaxNumSplits', 20); % Customize decision tree template
    model = fitcensemble(trainFeatures, trainLabels, 'Method', 'LogitBoost', 'NumLearningCycles', classifierParam, 'Learners', t, 'LearnRate', 0.1);
end

function model = trainKNN(trainFeatures, trainLabels, numNeighbors)
    % Train a KNN model
    model = fitcknn(trainFeatures, trainLabels, 'NumNeighbors', numNeighbors);
end
