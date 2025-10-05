import random


def train_test_split(X, y, test_size=0.2, random_state=None):
    """
    Split dataset into training and testing sets.
    
    This is crucial for evaluating ML models. We train on one portion of data
    and test on another unseen portion to estimate how well the model generalizes.
    
    Args:
        X (list): Features - each element is a sample
        y (list): Labels - corresponding target values
        test_size (float): Proportion of data for testing (0 to 1)
        random_state (int): Seed for reproducibility
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
        
    Why we split data:
        - Training set: Used to learn patterns
        - Test set: Used to evaluate generalization
        - Using the same data for both leads to overfitting (memorization)
        
    The golden rule:
        NEVER look at test data until final evaluation. Test set must remain
        completely unseen during model development.
    """
    if len(X) != len(y):
        raise ValueError("X and y must have same length")
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")
    
    if random_state is not None:
        random.seed(random_state)
    
    # Create indices and shuffle them
    indices = list(range(len(X)))
    random.shuffle(indices)
    
    # Calculate split point
    test_samples = int(len(X) * test_size)
    test_indices = indices[:test_samples]
    train_indices = indices[test_samples:]
    
    # Split data
    X_train = [X[i] for i in train_indices]
    X_test = [X[i] for i in test_indices]
    y_train = [y[i] for i in train_indices]
    y_test = [y[i] for i in test_indices]
    
    return X_train, X_test, y_train, y_test


def load_csv(filename, has_header=True):
    """
    Load data from a CSV file.
    
    Args:
        filename (str): Path to CSV file
        has_header (bool): Whether first row is header
        
    Returns:
        tuple: (data, headers) where data is list of lists
    """
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    if has_header:
        headers = lines[0].strip().split(',')
        data_lines = lines[1:]
    else:
        headers = None
        data_lines = lines
    
    data = []
    for line in data_lines:
        row = line.strip().split(',')
        # Try to convert to float, keep as string if not possible
        converted_row = []
        for value in row:
            try:
                converted_row.append(float(value))
            except ValueError:
                converted_row.append(value)
        data.append(converted_row)
    
    return data, headers