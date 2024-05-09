function model = trainMulticlassClassifier(trainFeatures, trainLabels, classifier, classifierParam)
    switch classifier
        case ''
        case 'random-forest'
            model = trainRandomForest(trainFeatures, trainLabels, classifierParam);
        case 'multiclass-svm'
            model = trainMulticlassSVM(trainFeatures, trainLabels, classifierParam, false);
        case 'multiclass-svm-weighted'
            model = trainMulticlassSVM(trainFeatures, trainLabels, classifierParam, true);
        case 'knn'
            model = trainKNN(trainFeatures, trainLabels, classifierParam);
        case 'mostFrequentDummy'
            frequencies = histcounts(labels(labels > 0));
            [~, model] = max(frequencies);
        case 'always1Dummy'
            % No training required for this classifier
        otherwise
            error('Unsupported classifier.')
    end
end

function model = trainMulticlassSVM(trainFeatures, trainLabels, boxConstraint, weighted)
    t = templateSVM('Standardize', true, 'KernelFunction', 'linear', ...
        'BoxConstraint', boxConstraint);

    if ~weighted
        model = fitcecoc(trainFeatures, trainLabels, 'Coding', 'onevsone', 'Learners', t);
    else
        % Train a SVM model for multiclass classification using fitcecoc with weighting
        model = fitcecoc(trainFeatures, trainLabels, 'Coding', 'onevsone', ...
            'Learners', t, 'Weights', calculateObservationWeights(trainLabels));
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

function model = trainRandomForest(trainFeatures, trainLabels,  classifierParam)
    % Train the Random Forest model
    model = TreeBagger(classifierParam, trainFeatures, trainLabels, 'Method', 'classification', 'OOBPrediction', 'On', 'MinLeafSize', 5, 'OOBPredictorImportance', 'On');
end

function model = trainKNN(trainFeatures, trainLabels, numNeighbors)
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
    model = fitcknn(trainFeatures, trainLabels, 'NumNeighbors', numNeighbors);
end
