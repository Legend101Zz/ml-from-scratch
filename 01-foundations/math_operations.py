def dot_product(vector_a,vector_b): 
    """
    Compute the dot product of two vectors.
    
    The dot product measures how much two vectors point in the same direction. 
    In machine learning, we use it to measure similarity between data points, compute predictions in linear models, and update parameters during training. 
    
    Args:
        vector_a (list): First vector 
        vector_b (list): Second vector 
    
    Returns:
        float: The dot product of the two vectors
    
    Example:
        >>> dot_product([1,2,3],[4,5,6])
        32 # This is 1*4 + 2*5 + 3*6
    """
    if len(vector_a) != len(vector_b):
        raise ValueError("Vectors must have the same length")
    
    result = 0
    for i in range(len(vector_a)):
        result += vector_a[i] * vector_b[i]
        
    return result

def matrix_multiply(matrix_a,matrix_b):
    """
    Multiply two matrics using the standard mathematical definition.
    
    Matrix multiplication is the foundation of most machine learning algorithms.
    Neural networks, linear regression, PCA, and many other techniques rely heavily on matrix operations.
    
    Args:
        matrix_a (list of lists) : First matrix ( m*n)
        matrix_b (list of lists) : Second matrix (n*p)
        
    Returns:
        list of lists : Result matrix (m*p)
    """
    # First we verify if multiplication is even possible
    rows_a , cols_a = len(matrix_a), len(matrix_a[0])
    rows_b, cols_b = len(matrix_b), len(matrix_b[0])
    
    if cols_a != rows_b:
        raise ValueError(f"Cannot multiply {rows_a}*{cols_b} and {rows_b}*{cols_b} matrices")
    
    # Initialise result matrix with zeros
    result = [[0 for _ in range(cols_b)] for _ in range(rows_a)]
    
    # Perform the actual multiplication
    for i in range(rows_a):
        for j in range(cols_b):
            # Each element of 𝐶 is a dot product between a row of 𝐴 and a column of 𝐵 , so
            column_b = [matrix_b[k][j] for k in range(rows_b)]
            result[i][j] = dot_product(rows_a[i],column_b) # notice how using the dot products forces us to two things :
            # 1) Think each row and column of a matrix as a vector
            # 2) Make the logic more easier to understand ( atleast for me it did)
    
    return result