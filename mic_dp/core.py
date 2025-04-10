"""
Core functionality for the mic_dp package.

This module provides functions for calculating feature relevance using different methods,
applying differential privacy with custom noise scaling, and evaluating the results.
"""

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import silhouette_score, adjusted_rand_score, v_measure_score
from sklearn.cluster import KMeans
from diffprivlib.mechanisms import Gaussian, Laplace
import math


def feature_selection(percentage, X, noise_scaling_factor):
    """
    Select features based on their noise scaling factors.
    
    Parameters
    ----------
    percentage : float
        Percentage of features to select (between 0 and 1)
    X : pandas.DataFrame
        Feature dataframe
    noise_scaling_factor : dict
        Dictionary mapping feature names to their noise scaling factors
        
    Returns
    -------
    list
        List of selected feature names
    """
    feature_num = int(percentage * X.shape[1])
    return sorted(noise_scaling_factor, key=noise_scaling_factor.get, reverse=True)[:feature_num]


def calculate_sensitivity(features):
    """
    Calculate sensitivity for each feature based on its range.
    
    Parameters
    ----------
    features : pandas.DataFrame
        Feature dataframe
        
    Returns
    -------
    dict
        Dictionary mapping feature names to their sensitivities
    """
    return {f: features[f].max() - features[f].min() for f in features.columns}


def calculate_mic_with_target(features, target):
    """
    Calculate Maximum Information Coefficient between features and target.
    
    Parameters
    ----------
    features : pandas.DataFrame
        Feature dataframe
    target : array-like
        Target variable
        
    Returns
    -------
    dict
        Dictionary mapping feature names to their MIC values
    """
    return {f: mutual_info_regression(features[f].values.reshape(-1, 1), target)[0] for f in features.columns}


def calculate_pearson_corr(features, target):
    """
    Calculate Pearson correlation between features and target.
    
    Parameters
    ----------
    features : pandas.DataFrame
        Feature dataframe
    target : array-like
        Target variable
        
    Returns
    -------
    dict
        Dictionary mapping feature names to their Pearson correlation values
    """
    target_series = pd.Series(target)
    return {f: features[f].corr(target_series, method="pearson") for f in features.columns}


def compute_mahalanobis_distances(features, target):
    """
    Compute Mahalanobis distances between features and target.
    
    Parameters
    ----------
    features : pandas.DataFrame
        Feature dataframe
    target : array-like
        Target variable
        
    Returns
    -------
    dict
        Dictionary mapping feature names to their Mahalanobis distances
    """
    distances = {}
    target = pd.Series(target, index=features.index)
    for col in features.columns:
        feat = features[col]
        mean_diff = feat.mean() - target.mean()
        var_diff = feat.var(ddof=1) + target.var(ddof=1) - 2 * feat.cov(target)
        distances[col] = np.inf if var_diff == 0 else np.sqrt((mean_diff ** 2) / var_diff)
    return distances


def normalize(values_dict, factor):
    """
    Normalize values in a dictionary to a specified range.
    
    Parameters
    ----------
    values_dict : dict
        Dictionary of values to normalize
    factor : float
        Scaling factor for normalization
        
    Returns
    -------
    dict
        Dictionary with normalized values
    """
    max_v, min_v = max(values_dict.values()), min(values_dict.values())
    return {f: (1 - (v - min_v) / (max_v - min_v)) * factor for f, v in values_dict.items()}


def noise_scaling_MIC(target, features, factor):
    """
    Calculate noise scaling factors based on Maximum Information Coefficient.
    
    Parameters
    ----------
    target : array-like
        Target variable
    features : pandas.DataFrame
        Feature dataframe
    factor : float
        Amplification factor for scaling
        
    Returns
    -------
    dict
        Dictionary mapping feature names to their MIC-based noise scaling factors
    """
    return normalize(calculate_mic_with_target(features, target), factor)


def noise_scaling_pearson(target, features, factor):
    """
    Calculate noise scaling factors based on Pearson correlation.
    
    Parameters
    ----------
    target : array-like
        Target variable
    features : pandas.DataFrame
        Feature dataframe
    factor : float
        Amplification factor for scaling
        
    Returns
    -------
    dict
        Dictionary mapping feature names to their Pearson-based noise scaling factors
    """
    return normalize(calculate_pearson_corr(features, target), factor)


def noise_scaling_mahalanobis_distances(target, features, factor):
    """
    Calculate noise scaling factors based on Mahalanobis distances.
    
    Parameters
    ----------
    target : array-like
        Target variable
    features : pandas.DataFrame
        Feature dataframe
    factor : float
        Amplification factor for scaling
        
    Returns
    -------
    dict
        Dictionary mapping feature names to their Mahalanobis-based noise scaling factors
    """
    dists = compute_mahalanobis_distances(features, target)
    normed = normalize(dists, factor)
    return {k: (factor if math.isnan(v) else v) for k, v in normed.items()}


def correlated_dp_gaussian(X, noise_factors, sensitivity, epsilon, delta):
    """
    Apply Gaussian differential privacy with custom noise scaling.
    
    Parameters
    ----------
    X : pandas.DataFrame
        Feature dataframe to privatize
    noise_factors : dict
        Dictionary mapping feature names to their noise scaling factors
    sensitivity : dict
        Dictionary mapping feature names to their sensitivities
    epsilon : float
        Privacy budget (ε)
    delta : float
        Privacy relaxation parameter (δ)
        
    Returns
    -------
    pandas.DataFrame
        Privatized feature dataframe
    """
    for f in X.columns:
        mech = Gaussian(epsilon=epsilon, delta=delta, sensitivity=sensitivity[f])
        X[f] = X[f].apply(lambda x: x + mech.randomise(x) * noise_factors[f])
    return X


def correlated_dp_laplace(X, noise_factors, sensitivity, epsilon, delta):
    """
    Apply Laplace differential privacy with custom noise scaling.
    
    Parameters
    ----------
    X : pandas.DataFrame
        Feature dataframe to privatize
    noise_factors : dict
        Dictionary mapping feature names to their noise scaling factors
    sensitivity : dict
        Dictionary mapping feature names to their sensitivities
    epsilon : float
        Privacy budget (ε)
    delta : float
        Privacy relaxation parameter (δ)
        
    Returns
    -------
    pandas.DataFrame
        Privatized feature dataframe
    """
    for f in X.columns:
        mech = Laplace(epsilon=epsilon, delta=delta, sensitivity=sensitivity[f])
        X[f] = X[f].apply(lambda x: x + mech.randomise(x) * noise_factors[f])
    return X


def mean_absolute_error(y_true, y_pred):
    """
    Calculate mean absolute error between true and predicted values.
    
    Parameters
    ----------
    y_true : array-like
        True values
    y_pred : array-like
        Predicted values
        
    Returns
    -------
    float
        Mean absolute error
    """
    return np.mean(np.abs(np.array(y_true) - np.array(y_pred)))


def cluster_and_evaluate(df, name, n_clusters):
    """
    Perform clustering and evaluate the results.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Data to cluster
    name : str
        Name of the clustering method
    n_clusters : int
        Number of clusters
        
    Returns
    -------
    tuple
        (silhouette_score, cluster_labels, kmeans_model)
    """
    model = KMeans(n_clusters=n_clusters, random_state=0)
    labels = model.fit_predict(df)
    score = silhouette_score(df, labels)
    return score, labels, model


def calculate_ari(labels1, labels2):
    """
    Calculate Adjusted Rand Index between two cluster labelings.
    
    Parameters
    ----------
    labels1 : array-like
        First set of cluster labels
    labels2 : array-like
        Second set of cluster labels
        
    Returns
    -------
    float
        Adjusted Rand Index
    """
    return adjusted_rand_score(labels1, labels2)


def calculate_v_measure(labels1, labels2):
    """
    Calculate V-measure between two cluster labelings.
    
    Parameters
    ----------
    labels1 : array-like
        First set of cluster labels
    labels2 : array-like
        Second set of cluster labels
        
    Returns
    -------
    float
        V-measure
    """
    return v_measure_score(labels1, labels2)
