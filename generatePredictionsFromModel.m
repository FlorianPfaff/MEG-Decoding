function predictions = generatePredictionsFromModel(testFeatures, model, classifier)
    switch classifier
        case 'mostFrequentDummy'
            predictions = model * ones(size(testFeatures, 1), 1);
        case 'always1Dummy'
            predictions = ones(size(testFeatures, 1), 1);
        case {'random-forest', 'multiclass-svm', 'multiclass-svm-weighted', 'knn'}
            [predictions, ~] = predict(model, testFeatures);
        otherwise
            error('Unsupported classifier.')
    end
    if iscell(predictions) && ischar(predictions{1})
        predictions = str2double(predictions);
    end
end