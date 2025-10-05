def min_max_normalize(data):
    """ 
    Normalize data to range [0,1] using min-max scaling.
    
    This transformation preserves the shape of the original distribution while 
    scaling all the values to the same range. Each value is transformed as:
    
    normalized_value = (value - min) / (max -min)
    
    Args:
        data (list): Original data values
        
    Returns:
        list: Normalized values between 0 and 1
    """
    if not data:
        raise ValueError("Cannot normalise empty data")
    
    min_val = min(data)
    max_val = max(data)
    
    if max_val == min_val:
        # hence all values are same , return array of 0.5
        return [0.5 for _ in data]
    range_val = max_val - min_val
    return [(x-min_val)/range_val for x in data]

def z_score_normalize(data):
    """ 
    Normalize data using z-score standardization (mean=0, std=1)
    This transformation centers the data around 0 and scales it so the standard
    deviation is 1. Each value is transformed as:
    
    z_score = (value - mean) / std_dev
    
    Args:
        data (list): Original data values
        
    Returns:
        list: Standardized values with mean≈0 and std≈1
    """
    from ..foundations.statistics import mean, standard_deviation
    
    if not data:
        raise ValueError("Cannot standardize empty data")
    
    data_mean = mean(data)
    data_std = standard_deviation(data)
    
    if data_std == 0:
        # all values are same , so return array of zeros:
        return [0.0 for _ in data]
    
    return [ (x-data_mean)/data_std for x in data]

def normalize_features(dataset, method='z-score'):
    """ 
    Normalize all the features of a dataset.
    
    Args:
        dataset (list of lists): Each inner list is a sample, each element is a feature
                                [[sample1_feature1, sample1_feature2, ...],
                                 [sample2_feature1, sample2_feature2, ...], ...]
        method (str): 'z-score' or 'min-max'
        
    Returns:
        list of lists: Normalized dataset
        
    Important note:
        In real ML, you must calculate normalization parameters (mean, std, min, max)
        from TRAINING data only, then apply them to test data. This prevents
        "data leakage" where test data information influences training.
    """
    if not dataset or not dataset[0]:
        raise ValueError("Dataset cannot be empty")
    
    n_samples = len(dataset)
    n_features = len(dataset[0])
    
    # First we transpose the matrix to have features in rows 
    features = [[dataset[i][j] for i in range(n_samples)] for j in range(n_features)]
    
    # Normalize each feature
    if method == 'z-score':
        normalized_features = [z_score_normalize(feature) for feature in features]
    elif method == 'min-max':
        normalized_features = [min_max_normalize(feature) for feature in features]
    else:
        raise ValueError("Method must be 'z-score' or 'min-max'")
    
    # Transpose back to original shape
    normalized_dataset = [[normalized_features[j][i] for j in range(n_features)] 
                          for i in range(n_samples)]
    
    return normalized_dataset