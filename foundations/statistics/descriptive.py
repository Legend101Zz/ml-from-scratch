"""
Why this module exists:
-----------------------
Statistics is the language of data. Before we can do machine learning,
we need to understand our data: Where is the center? How spread out is it?
How do variables relate to each other?

Every ML algorithm uses these fundamental statistical concepts:
- Linear Regression: minimizes mean squared error
- Normalization: uses mean and standard deviation
- Gradient Descent: computes mean gradient across samples
- PCA: uses covariance matrices

This module implements statistics WITHOUT numpy, using only math module!
"""

import math


def mean(data):
    """
    Calculate the arithmetic mean (average) of a dataset
    
    The mean represents the central tendency of data and appears everywhere in ML.
    Many algorithms assume data is centered around its mean, and we often normalize data by
    subtracting the mean.
    
    Args: 
        data (list) : List of numerical values
        
    Returns:
        float: The arithmetic mean of the data
    """
    if not data:
        raise ValueError("Cannot calculate mean of empty dataset")
    return sum(data) / len(data)

def median(data):
    """
    Compute the median (middle value) of a dataset.
    
    Mathematical Definition:
    ------------------------
    For sorted data x_1 <= x_2 <= ... <= x_n:
    
    - If n is odd:  median = x_{(n+1)/2}
    - If n is even: median = (x_{n/2} + x_{n/2 + 1}) / 2
    
    Intuition:
    ----------
    The "middle" value that splits data into two equal halves.
    
    Think of people lined up by height - the median is the person in the middle.
    Half are shorter, half are taller.
    
    Properties:
    -----------
    - Robust to outliers (unlike mean!)
    - Better for skewed distributions
    - Represents the "typical" value
    
    Parameters:
    -----------
    data : list of float/int
        The dataset (must not be empty)
    
    Returns:
    --------
    float : The median value
    
    Example:
    --------
    >>> median([1, 2, 3, 4, 5])
    3
    >>> median([1, 2, 100, 200])  # Mean would be 75.75, median is 51
    51.0
    """
    if not data:
        raise ValueError("Cannot compute median of empty dataset")
    
    sorted_data = sorted(data)
    n = len(sorted_data)
    
    if n%2 == 1:
        # Odd length: return middle element
        middle_index = n // 2
        return float(sorted_data[middle_index])
    else:
        # Even length: return avg of two middle elements
        middle1 = n // 2
        middle2 = n // 2 - 1
        return (sorted_data[middle1] + sorted_data[middle2]) / 2.0

def variance(data, sample=True):
    """
    Calculate the variance of a dataset
    
    Variance measures how spread out the data points are from the mean .
    Understanding variance is crucial for machine learning because many algorithms are sensitive to the scale and spread of input features,
    
    Args:
        data (list): List of numerical values
        sample (bool): If True , calculate sample variance ( divide by n-1)
                        If False, calculate population variance ( divide by n)
        Returns:
            float: The variance of the data
    """
    if len(data) < 2 :
        raise ValueError("Need at least 2 values to calculate variance")
    data_mean = mean(data)
    squared_differences = [(x - data_mean) ** 2 for x in data]
    
    divisor = len(data) - 1 if sample else len(data)
    return sum(squared_differences) /divisor

def standard_deviation(data,sample=True):
    """
    Calculate the standard deviation of a dataset .
    
    Standard deviation is the square root of variance and gives us a measure of spread in the same units as original data .
    This makes it more interpretable than variance for understanding data distributions. 
    """
    return variance(data,sample) ** 0.5

def mean_absolute_deviation(data):
    """
    Compute Mean Absolute Deviation (MAD) - average distance from mean.
    
    Mathematical Definition:
    ------------------------
    MAD = (1/n) * sum_{i=1}^{n} |x_i - mean(x)|
    
    Intuition:
    ----------
    Like standard deviation, but using absolute value instead of squaring.
    
    Differences from standard deviation:
    - MAD: uses |deviation| - treats all deviations equally
    - Std: uses deviation² - penalizes large deviations more
    
    MAD is more robust to outliers than standard deviation!
    
    Parameters:
    -----------
    data : list of float/int
        The dataset
    
    Returns:
    --------
    float : The mean absolute deviation
    
    Example:
    --------
    >>> mean_absolute_deviation([1, 2, 3, 4, 5])
    1.2
    """
    if not data:
        raise ValueError("Cannot compute MAD of empty dataset")
    
    mu = mean(data)
    
    sum_abs_deviations = 0.0
    for x in data:
        sum_abs_deviations += abs(x - mu)
    
    return sum_abs_deviations / len(data)

def covariance(x,y):
    """ 
    Calculate the covariance between two variables.
    
    Covariance measures how two variables change together. If x increases when y increases, covariance is positive. If x increases when y decreases, covariance is negative. If they don't have a linear relationship, covariance is near zero.
    
    Formula: Cov(X,Y) = Σ((xi - x̄)(yi - ȳ)) / (n-1)
    
    Args:
        x (list): First variable
        y (list): Second variable
        
    Returns:
        float: Covariance between x and y
    """
    if len(x) != len(y):
        raise ValueError("Variables must be of same length")
    if len(x) < 2 :
        raise ValueError("Need at least 2 data points")
    
    mean_x = mean(x)
    mean_y = mean(y)
    
    # Now we calculate sum of products of deviations
    covariance_sum = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(len(x)))
    
    return covariance_sum/ (len(x) -1)

def correlation(x,y):
    """ 
    Calculate the Pearson correlation coefficient between two variables
    
    Correlation is like covariance but normalized to always be between -1 and 1.
    This makes it easier to interpret:
    - +1: perfect positive linear relationship
    -  0: no linear relationship
    - -1: perfect negative linear relationship
    
    Formula: r = Cov(X,Y) / (σx * σy)
    
    Args:
        x (list): First variable
        y (list): Second variable
        
    Returns:
        float: Correlation coefficient between -1 and 1
        
    Interpretation guide:
        0.0 - 0.3: Weak relationship
        0.3 - 0.7: Moderate relationship
        0.7 - 1.0: Strong relationship
    """
    cov = covariance(x,y)
    std_x = standard_deviation(x)
    std_y = standard_deviation(y)
    
    if std_x == 0 or std_y == y:
        raise ValueError("Cannot calculate correlation when std is zero" )  
    return cov / (std_x * std_y)

def covariance_matrix(data):
    """ 
    Calculate the covariance matrix for a dataset with multiple features.
    
    The covariance matrix shows how every pair of feature covaries with each other.
    This matrix is central to many machine learning algorithms including PCA,
    linear discriminant analysis, and multivariate Gaussian distributions.
    
    Args:
        data (list of lists): Each inner list is a feature, each position is a sample
                             [[feature1_values], [feature2_values], ...]
        
    Returns:
        list of lists: Covariance matrix where element [i][j] is cov(feature_i, feature_j)
        
    Properties of covariance matrix:
        - Diagonal elements are variances of each feature
        - Off-diagonal elements are covariances between feature pairs
        - Matrix is symmetric: cov(X,Y) = cov(Y,X)
        - Positive semi-definite (important mathematical property)
    """
    n_features = len(data)
    cov_matrix= [[0 for _ in range(n_features)] for _ in range(n_features)]
    
    for i in range(n_features):
        for j in range(n_features):
            if i == j:
                # Diagonal: variance of feature i
                cov_matrix[i][j] = variance(data[i])
            else: 
                # Off-diagonal : covariance between i and j
                cov_matrix[i][j] = covariance(data[i], data[j]) 
                
    return cov_matrix

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================

def describe(data):
    """
    Compute comprehensive summary statistics for a dataset.
    
    This function gives you a complete picture of your data at a glance!
    
    Returns a dictionary with:
    - count: number of data points
    - mean: average value
    - median: middle value
    - std: standard deviation
    - var: variance
    - min: smallest value
    - max: largest value
    - range: max - min
    
    Parameters:
    -----------
    data : list of float/int
        The dataset to describe
    
    Returns:
    --------
    dict : Dictionary containing all summary statistics
    
    Example:
    --------
    >>> describe([1, 2, 3, 4, 5, 100])
    {
        'count': 6,
        'mean': 19.166...,
        'median': 3.5,
        'std': 39.67...,
        ...
    }
    """
    if not data:
        raise ValueError("Cannot describe empty dataset")
    
    return {
        'count': len(data),
        'mean': mean(data),
        'median': median(data),
        'std': standard_deviation(data),
        'var': variance(data),
        'min': min(data),
        'max': max(data),
        'range': max(data) - min(data),
        'mad': mean_absolute_deviation(data)
    }