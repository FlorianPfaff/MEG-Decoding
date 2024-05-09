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