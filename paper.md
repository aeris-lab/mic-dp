---
title: 'mic_dp: A Python package for maximum information coefficient differential privacy'
tags:
  - Python
  - differential privacy
  - feature selection
  - mutual information
  - machine learning
  - privacy-preserving
authors:
  - name: Wenjun Yang
    orcid: 0000-0000-0000-0000
    affiliation: 1
  - name: Eyhab Al-masri
    orcid: 0000-0000-0000-0000
    affiliation: 1
  - name: Olivera Kotevska
    orcid: 0000-0000-0000-0000
    affiliation: 2
affiliations:
  - name: University of Washington Tacoma, United States
    index: 1
  - name: Oak Ridge National Laboratory, United States
    index: 2
date: 10 April 2025
bibliography: paper.bib
---

# Summary

`mic_dp` is a Python package that enables differentially private data transformation guided by the *Maximum Information Coefficient* (MIC), with application to both supervised and unsupervised learning tasks. Traditional differential privacy (DP) mechanisms often degrade utility uniformly across features. In contrast, `mic_dp` uses MIC to scale the noise injection, preserving more utility in informative features.

This package includes functions for:
- Calculating MIC, Pearson, and Mahalanobis-based feature relevance
- Feature selection based on scaled importance
- Applying Gaussian or Laplace DP mechanisms using custom noise scaling
- Evaluating MAE, clustering scores, and plotting results

The library has been evaluated using:
1. **ACI Dataset (Adult Census Income)** from the UCI repository for supervised learning [@Dua2019]. This dataset includes demographic features used to predict income classes.
2. **Household Electricity Demand (HED)** dataset from the 2009 Midwest RECS dataset for unsupervised learning [@EIA2009]. This time series dataset provides daily electricity consumption profiles across households.

Our experiments show that MIC-guided DP mechanisms consistently outperform Pearson, Mahalanobis, and baseline DP in terms of feature and prediction accuracy under privacy constraints. In unsupervised settings, MIC-DP preserves cluster structures better, as shown by silhouette score, ARI, and V-measure.

# Statement of need

There is a growing demand for privacy-preserving data analysis tools that can maintain high utility. While several differential privacy libraries exist (e.g., diffprivlib [@Holohan2019]), few provide support for custom noise scaling based on statistical relevance like MIC. `mic_dp` fills this gap by providing a framework to perform smart, feature-sensitive privacy transformations and rigorous evaluations.

The Maximum Information Coefficient (MIC) [@Reshef2011] is a measure of the strength of the linear or non-linear association between two variables. Unlike traditional correlation measures, MIC can detect a wide range of associations, making it particularly valuable for identifying informative features in complex datasets. By leveraging MIC to guide differential privacy mechanisms, our package enables more effective privacy-utility trade-offs than uniform noise approaches.

Researchers and practitioners in fields such as healthcare, finance, and social sciences can use `mic_dp` to:

1. Apply differential privacy while preserving analytical utility
2. Conduct feature selection under privacy constraints
3. Compare different noise-scaling strategies
4. Evaluate the impact of privacy on supervised and unsupervised learning tasks

# Implementation

The `mic_dp` package is implemented in Python and builds upon established libraries including scikit-learn [@Pedregosa2011] for machine learning functionality and IBM's diffprivlib [@Holohan2019] for differential privacy mechanisms. The core functionality includes:

```python
# Calculate MIC-based noise scaling factors
noise_factors = noise_scaling_MIC(target, features, amplification_factor)

# Calculate sensitivity for each feature
sensitivity = calculate_sensitivity(features)

# Apply Gaussian DP with MIC-guided noise scaling
private_data = correlated_dp_gaussian(
    features.copy(), 
    noise_factors, 
    sensitivity, 
    epsilon=0.5, 
    delta=1e-5
)
```

The package also provides utilities for evaluating the impact of privacy on model performance:

```python
# Measure feature distortion
mae = mean_absolute_error(original_features, private_features)

# Evaluate clustering quality
silhouette, labels, model = cluster_and_evaluate(private_data, "MIC-DP", n_clusters=3)
```

# Example usage

The following example demonstrates how to apply MIC-guided differential privacy to a supervised learning task:

```python
from mic_dp.core import (
    noise_scaling_MIC, calculate_sensitivity, 
    correlated_dp_gaussian, mean_absolute_error
)
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load and preprocess data
df = pd.read_csv('adult.csv')
X = df.select_dtypes(include=['number'])
X_norm = pd.DataFrame(MinMaxScaler().fit_transform(X), columns=X.columns)
y = df['income'].astype('category').cat.codes

# Apply MIC-guided differential privacy
noise_factors = noise_scaling_MIC(y, X_norm, amplification_factor=5)
sensitivity = calculate_sensitivity(X_norm)
private_X = correlated_dp_gaussian(X_norm.copy(), noise_factors, sensitivity, epsilon=0.5, delta=1e-5)

# Evaluate utility
feature_mae = mean_absolute_error(X_norm, private_X)
print(f"Feature MAE: {feature_mae:.4f}")

# Train model on private data
model = RandomForestRegressor()
model.fit(private_X, y)
predictions = model.predict(private_X)
prediction_mae = mean_absolute_error(y, predictions)
print(f"Prediction MAE: {prediction_mae:.4f}")
```

# Acknowledgements

We acknowledge the creators of the ACI and HED datasets for making their data publicly available. We also thank the developers of scikit-learn and diffprivlib for their valuable tools that enabled this work.

# References
