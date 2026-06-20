function model = trainMulticlassClassifier(trainFeatures, trainLabels, classifier, classifierParam)
    switch classifier
        case ''
        case 'random-forest'
            model = trainRandomForest(trainFeatures, trainLabels, classifierParam);
        case 'multiclass-svm'
            model = trainMulticlassSVM(trainFeatures, trainLabels, classifierParam, false);
        case 'multiclass-svm-weighted'
            model = trainMulticlassSVM(trainFeatures, trainLabels, classifierParam, true);
        case 'branch-fusion-svm'
            model = trainBranchFusionSVM(trainFeatures, trainLabels, classifierParam, false);
        case 'branch-fusion-svm-weighted'
            model = trainBranchFusionSVM(trainFeatures, trainLabels, classifierParam, true);
        case 'knn'
            model = trainKNN(trainFeatures, trainLabels, classifierParam);
        case 'mostFrequentDummy'
            frequencies = histcounts(trainLabels(trainLabels > 0));
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
end

function model = trainBranchFusionSVM(trainFeaturesByBranch, trainLabels, boxConstraint, weighted)
    % Train one linear ECOC SVM per feature branch. At prediction time, the
    % per-branch class scores are summed. This implements a lightweight
    % parallel temporal/spatial branch model without adding a deep-learning
    % dependency and without using any held-out target data for fitting.
    assert(iscell(trainFeaturesByBranch), 'Branch-fusion classifiers expect a cell array of branch features.')
    assert(~isempty(trainFeaturesByBranch), 'At least one branch is required for branch-fusion classification.')

    nBranches = numel(trainFeaturesByBranch);
    model = struct();
    model.branchModels = cell(1, nBranches);
    model.classNames = [];
    model.fusion = 'sum-score';

    for b = 1:nBranches
        assert(size(trainFeaturesByBranch{b}, 1) == numel(trainLabels), ...
            'All branch feature matrices must have one row per training label.')
        model.branchModels{b} = trainMulticlassSVM(trainFeaturesByBranch{b}, trainLabels, boxConstraint, weighted);
        if b == 1
            model.classNames = model.branchModels{b}.ClassNames;
        else
            assert(isequal(model.classNames, model.branchModels{b}.ClassNames), ...
                'All branch models must use the same class order.')
        end
    end
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