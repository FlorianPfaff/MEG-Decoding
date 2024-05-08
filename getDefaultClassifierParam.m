function classifierParam = getDefaultClassifierParam(classifier)
    switch classifier
        case 'lasso'
            classifierParam = 0.005;
        case {'multiclass-svm', 'multiclass-svm-weighted', 'binary-svm'}
            classifierParam = 0.5;
        case {'random-forest', 'gradient-boosting'}
            classifierParam = 100;
        case 'knn'
            classifierParam = 5;
        case {'mostFrequentDummy', 'always1Dummy'}
            classifierParam = [];
        otherwise
            error('No default parameter for the classifier.')
    end
end