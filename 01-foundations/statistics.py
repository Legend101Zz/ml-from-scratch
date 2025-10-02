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