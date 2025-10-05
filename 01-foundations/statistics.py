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