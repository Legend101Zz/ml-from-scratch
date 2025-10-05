def euclidean_distance(point1, point2):
    """
    Calculate the Euclidean (straight-line) distance between two points.
    
    This is the most intuitive distance metric - it's the straight-line distance
    you would measure with a ruler. In 2D, it's the Pythagorean theorem.
    
    Formula: √(Σ(xi - yi)²)
    
    Args:
        point1 (list): First point coordinates [x1, x2, ..., xn]
        point2 (list): Second point coordinates [y1, y2, ..., yn]
        
    Returns:
        float: Euclidean distance between the points
        
    Geometric Intuition:
        In 2D: If point1 = [0, 0] and point2 = [3, 4]
        Distance = √((3-0)² + (4-0)²) = √(9 + 16) = √25 = 5
        
        This forms a right triangle with legs 3 and 4, hypotenuse 5.
        
    In ML, we use this to:
        - Find similar data points (k-NN)
        - Measure prediction error (often squared Euclidean distance)
        - Define clusters in clustering algorithms
    """
    if len(point1) != len(point2):
        raise ValueError("Points must have same number of dimensions")
    
    squared_differences = [(point1[i] - point2[i]) ** 2 for i in range(len(point1))]
    return sum(squared_differences) ** 0.5


def manhattan_distance(point1, point2):
    """
    Calculate the Manhattan (city-block) distance between two points.
    
    Named after Manhattan's grid layout, this measures distance as if you can
    only move along grid lines (like walking through city blocks). It's the sum
    of absolute differences in each dimension.
    
    Formula: Σ|xi - yi|
    
    Args:
        point1 (list): First point coordinates
        point2 (list): Second point coordinates
        
    Returns:
        float: Manhattan distance between points
        
    When to use Manhattan vs Euclidean:
        - Manhattan is less sensitive to outliers (no squaring)
        - Manhattan is faster to compute (no square root)
        - Euclidean is better when direction matters
        - Manhattan is better for high-dimensional sparse data
        
    Example:
        Point1 = [0, 0], Point2 = [3, 4]
        Manhattan distance = |3-0| + |4-0| = 3 + 4 = 7
        (vs Euclidean distance of 5)
    """
    if len(point1) != len(point2):
        raise ValueError("Points must have same number of dimensions")
    
    absolute_differences = [abs(point1[i] - point2[i]) for i in range(len(point1))]
    return sum(absolute_differences)


def minkowski_distance(point1, point2, p=2):
    """
    Calculate the Minkowski distance - a generalization of Euclidean and Manhattan.
    
    This is a family of distance metrics controlled by parameter p:
    - p=1: Manhattan distance
    - p=2: Euclidean distance
    - p=∞: Chebyshev distance (maximum difference in any dimension)
    
    Formula: (Σ|xi - yi|^p)^(1/p)
    
    Args:
        point1 (list): First point
        point2 (list): Second point
        p (float): Order of the norm (p ≥ 1)
        
    Returns:
        float: Minkowski distance
        
    Why this generalization matters:
        Different values of p emphasize different aspects of dissimilarity.
        Higher p values give more weight to large differences in any single dimension.
    """
    if len(point1) != len(point2):
        raise ValueError("Points must have same number of dimensions")
    if p < 1:
        raise ValueError("p must be at least 1")
    
    powered_differences = [abs(point1[i] - point2[i]) ** p for i in range(len(point1))]
    return sum(powered_differences) ** (1/p)


def cosine_similarity(vector1, vector2):
    """
    Calculate cosine similarity between two vectors.
    
    Unlike distance metrics, cosine similarity measures the angle between vectors
    rather than their magnitude. This makes it useful when the direction matters
    more than the absolute values.
    
    Formula: (v1 · v2) / (||v1|| × ||v2||)
    
    Returns value between -1 and 1:
    - 1: Vectors point in exactly the same direction
    - 0: Vectors are perpendicular (orthogonal)
    - -1: Vectors point in opposite directions
    
    Args:
        vector1 (list): First vector
        vector2 (list): Second vector
        
    Returns:
        float: Cosine similarity (-1 to 1)
        
    Common uses in ML:
        - Text similarity (document vectors)
        - Recommendation systems (user preference vectors)
        - Face recognition (feature vectors)
        
    Why it's useful:
        Imagine two documents with identical word ratios but different lengths.
        Euclidean distance would be large, but cosine similarity would be 1,
        correctly identifying them as similar.
    """
    if len(vector1) != len(vector2):
        raise ValueError("Vectors must have same length")
    
    # Import dot_product from math_operations
    from math_operations import dot_product
    
    dot_prod = dot_product(vector1, vector2)
    
    # Calculate magnitudes (L2 norms)
    magnitude1 = sum(x**2 for x in vector1) ** 0.5
    magnitude2 = sum(x**2 for x in vector2) ** 0.5
    
    if magnitude1 == 0 or magnitude2 == 0:
        raise ValueError("Cannot calculate cosine similarity with zero vector")
    
    return dot_prod / (magnitude1 * magnitude2)


def cosine_distance(vector1, vector2):
    """
    Convert cosine similarity to a distance metric.
    
    Since similarity of 1 means identical and 0 means orthogonal,
    we define distance as 1 - similarity.
    
    Returns value between 0 and 2:
    - 0: Identical direction
    - 1: Perpendicular
    - 2: Opposite directions
    """
    return 1 - cosine_similarity(vector1, vector2)