function predictions = generatePredictions(testFeatures, model, classifier)
    switch classifier
        case 'mostFrequentDummy'
            predictions = predictDummyClassifier(model, testFeatures);
        case 'always1Dummy'
            predictions = model(testFeatures);
        case 'random-forest'
            predictions = predictRandomForest(model, testFeatures);
        case 'gradient-boosting'
            predictions = predictGradientBoosting(model, testFeatures);
        case 'multiclass-svm'
            predictions = predictMulticlassSVM(model, testFeatures);
        case 'multiclass-svm-weighted'
            predictions = predictMulticlassSVM(model, testFeatures);
        case 'knn'
            predictions = predictKNN(model, testFeatures);
        case 'lasso'
            predictions = predictLassoGLM(model, testFeatures);
        case 'svm-binary'
            predictions = predictBinarySVM(model, testFeatures);
        otherwise
            error('Invalid classifier type');
    end
end

function predictions = predictDummyClassifier(model, testFeatures)
    predictions = model(testFeatures);
end


function predictions = predictLassoGLM(model, testFeatures)
    % Predict using the Lasso GLM model
    predictions = glmval([model.intercept; model.beta], testFeatures, 'logit');
end


function predictions = predictBinarySVM(model, testFeatures)
    % Predict using the binary SVM model
    [~, score] = predict(model, testFeatures);
    predictions = score(:, 2); % The second column contains scores for the positive class
end

function predictions = predictMulticlassSVM(model, testFeatures)
    % Predict using the multiclass SVM model
    predictions = predict(model, testFeatures);
end


function predictions = predictRandomForest(model, testFeatures)
    % Predict using the Random Forest model
    [predictions, ~] = predict(model, testFeatures);
    % Convert predictions from cell to numeric, as predict returns cell array for classification
    predictions = str2double(predictions);
end

function predictions = predictGradientBoosting(model, testFeatures)
    % Predict using the Gradient Boosting model
    predictions = predict(model, testFeatures);
end


function predictions = predictKNN(model, testFeatures)
    % Predict using the KNN model
    predictions = predict(model, testFeatures);
end