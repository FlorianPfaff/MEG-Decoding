function predictions = generatePredictionsFromModel(testFeatures, model, classifier)
    switch classifier
        case 'mostFrequentDummy'
            predictions = model * ones(size(testFeatures, 1), 1);
        case 'always1Dummy'
            predictions = ones(size(testFeatures, 1), 1);
        case {'random-forest', 'multiclass-svm', 'multiclass-svm-weighted', 'knn'}
            [predictions, ~] = predict(model, testFeatures);
        case {'branch-fusion-svm', 'branch-fusion-svm-weighted'}
            predictions = generateBranchFusionPredictions(testFeatures, model);
        otherwise
            error('Unsupported classifier.')
    end
    if iscell(predictions) && ischar(predictions{1})
        predictions = str2double(predictions);
    end
end

function predictions = generateBranchFusionPredictions(testFeaturesByBranch, model)
    assert(iscell(testFeaturesByBranch), 'Branch-fusion prediction expects a cell array of branch features.')
    assert(numel(testFeaturesByBranch) == numel(model.branchModels), ...
        'The number of test branches must match the trained branch-fusion model.')

    classNames = model.classNames(:);
    nClasses = numel(classNames);
    nTest = size(testFeaturesByBranch{1}, 1);
    scoreSum = zeros(nTest, nClasses);

    for b = 1:numel(model.branchModels)
        assert(size(testFeaturesByBranch{b}, 1) == nTest, ...
            'All test branch matrices must have the same number of rows.')

        [~, branchScores] = predict(model.branchModels{b}, testFeaturesByBranch{b});
        branchClassNames = model.branchModels{b}.ClassNames(:);
        [isKnownClass, classIndices] = ismember(branchClassNames, classNames);
        assert(all(isKnownClass), 'A branch model returned an unknown class label.')

        scoreSum(:, classIndices) = scoreSum(:, classIndices) + branchScores;
    end

    [~, bestClassIndex] = max(scoreSum, [], 2);
    predictions = classNames(bestClassIndex);
end