"""
POLYNOMIAL FEATURES: Making Linear Models Fit Curves!
=====================================================

This is where my mind was blown: You can fit complex curves using LINEAR regression!

THE BIG INSIGHT:
---------------
I used to think "linear regression" meant you could only fit straight lines. Wrong!
"Linear" refers to being linear in the PARAMETERS (weights), not linear in the FEATURES.

This means you can transform features in non-linear ways, then use linear regression:

Original features:  x = [x₁]
Transform to:       x' = [1, x₁, x₁²]
Fit linear model:   y = w₀·1 + w₁·x₁ + w₂·x₁²

The model is linear in [w₀, w₁, w₂] but fits a PARABOLA in x₁!

This generalizes to any transformation:
- Polynomials: [x, x², x³, ...]
- Trigonometric: [sin(x), cos(x), ...]
- Logarithmic: [log(x), √x, ...]
- Interactions: [x₁·x₂, x₁·x₃, ...]

POLYNOMIAL FEATURES SPECIFICALLY:
--------------------------------
For degree d, we generate all polynomial terms up to that degree:

degree=1: [1, x]
degree=2: [1, x, x²]
degree=3: [1, x, x², x³]

For multiple features [x₁, x₂], degree=2:
[1, x₁, x₂, x₁², x₁x₂, x₂²]

This is called the "polynomial basis expansion."

THE WARNING:
-----------
High degrees can cause OVERFITTING! With degree=10, you can fit any 11 points
perfectly, but the curve will be crazy between those points. The model memorizes
the training data instead of learning the pattern.

Rule of thumb: Start with degree=2 or 3, increase only if necessary.

MY LEARNING JOURNEY:
-------------------
When I first implemented this, I was amazed that the SAME linear regression code
could suddenly fit curves! The secret is feature engineering — transforming your
inputs before feeding them to the model.

This taught me a fundamental lesson: The distinction between "linear" and "non-linear"
models is less clear than I thought. Any model can become more expressive through
clever feature transformations!
"""

from foundations.linear_algebra.vectors_and_matrices import Matrix, Vector
from typing import List, Tuple


class PolynomialFeatures:
    """
    Generate polynomial features up to a given degree.
    
    This transformer takes input features and creates polynomial combinations,
    allowing linear models to fit non-linear patterns!
    
    MATHEMATICAL DEFINITION:
    -----------------------
    Given input features x = [x₁, x₂, ..., xₙ] and degree d, generate all
    monomials (products of variables) with total degree ≤ d:
    
    degree=1: [1, x₁, x₂, ..., xₙ]
    degree=2: [1, x₁, x₂, ..., xₙ, x₁², x₁x₂, ..., x₂², ..., xₙ²]
    degree=3: [1, x₁, x₂, ..., xₙ, x₁², x₁x₂, ..., x₁³, x₁²x₂, ..., xₙ³]
    
    THE COMBINATORIAL EXPLOSION:
    ----------------------------
    The number of features grows quickly:
    - 1 feature, degree d: d+1 features
    - 2 features, degree 2: 6 features (1, x₁, x₂, x₁², x₁x₂, x₂²)
    - 3 features, degree 2: 10 features
    - 10 features, degree 3: 286 features!
    
    Formula: C(n+d, d) where n=num_features, d=degree
    
    This explosion is why we usually keep degree ≤ 3 in practice.
    
    EXAMPLE TRANSFORMATIONS:
    -----------------------
    Input: [[2], [3], [4]]  (single feature x)
    degree=2:
    Output: [[1, 2, 4],     (1, x, x²)
             [1, 3, 9],
             [1, 4, 16]]
    
    Input: [[2, 3], [4, 5]]  (two features x₁, x₂)
    degree=2:
    Output: [[1, 2, 3, 4, 6, 9],     (1, x₁, x₂, x₁², x₁x₂, x₂²)
             [1, 4, 5, 16, 20, 25]]
    
    INTERACTION TERMS:
    -----------------
    Polynomial features include interaction terms like x₁·x₂. These capture
    how features work together!
    
    Example: Predicting house price from [sqft, location_quality]
    - x₁ (sqft): Direct effect of size
    - x₂ (location): Direct effect of location
    - x₁·x₂: Size matters more in good locations!
    
    The model learns: price = w₀ + w₁·sqft + w₂·location + w₃·sqft·location
    
    INCLUDE_BIAS PARAMETER:
    ----------------------
    If include_bias=True (default), we add a column of ones.
    If include_bias=False, we assume X already has a bias column.
    
    Most of the time, use include_bias=True for convenience.
    
    PARAMETERS:
    ----------
    degree : int, default=2
        Maximum degree of polynomial features.
        degree=1: Just adds bias (no transformation)
        degree=2: Squares and pairwise products
        degree=3: Cubes and three-way products
        
    include_bias : bool, default=True
        Whether to include a bias column (column of ones).
        
    ATTRIBUTES:
    ----------
    degree : int
        The degree parameter
        
    n_input_features_ : int
        Number of features in the input (set during fit)
        
    n_output_features_ : int
        Number of features in the output (set during fit)
        
    feature_names_ : List[str]
        Names describing each output feature (for debugging)
        
    EXAMPLE USAGE:
    -------------
    >>> X = Matrix([[2], [3], [4]])  # Single feature
    >>> 
    >>> poly = PolynomialFeatures(degree=2)
    >>> X_poly = poly.fit_transform(X)
    >>> # X_poly is now [[1, 2, 4], [1, 3, 9], [1, 4, 16]]
    >>> 
    >>> # Now train linear regression on X_poly to fit a parabola!
    >>> model = LinearRegression()
    >>> model.fit(X_poly, y)
    """
    def __init__(self, degree: int = 2, include_bias: bool = True):
        if degree < 1:
            raise ValueError(f"degree must be >= 1, got {degree}")
        
        self.degree = degree
        self.include_bias = include_bias
        
        # These will be set during fit()
        self.n_input_features_ = None
        self.n_output_features_ = None
        self.feature_names_ = None
    
    def fit(self, X: Matrix) -> 'PolynomialFeatures':
        """
        Compute number of output features and feature names.
        
        This doesn't actually transform the data, it just figures out what the
        transformation will look like (for consistency with scikit-learn API).
        
        PARAMETERS:
        ----------
        X : Matrix, shape (n_samples, n_features)
            Input features
            
        RETURNS:
        -------
        self : PolynomialFeatures
            Returns self for method chaining
        """
        
        self.n_input_features_ = X.num_cols
        
        # Compute the number of output features
        # Formula: C(n+d, d) where n=num_features, d=degree
        # But we'll compute it by actually generating the combinations
        
        powers = self._generate_powers()
        self.n_output_features_ = len(powers)
        
        # Generate feature names for debugging/inspection
        self.feature_names_ = self._generate_feature_names(powers)
        
        return self
    
    def transform(self, X: Matrix) -> Matrix:
        """
        Transform data to polynomial features.
        
        This is where the magic happens! We take each sample and create all
        polynomial combinations up to the specified degree.
        
        THE ALGORITHM:
        -------------
        For each sample [x₁, x₂, ..., xₙ]:
        1. Generate all combinations of powers: (p₁, p₂, ..., pₙ) where Σpᵢ ≤ degree
        2. For each combination, compute: x₁^p₁ · x₂^p₂ · ... · xₙ^pₙ
        3. Collect all these products as the new feature vector
        
        EXAMPLE:
        -------
        Input: x = [2, 3], degree=2
        
        Powers to generate: (0,0), (1,0), (0,1), (2,0), (1,1), (0,2)
        Corresponding features:
        - (0,0): 2⁰·3⁰ = 1        (bias)
        - (1,0): 2¹·3⁰ = 2        (x₁)
        - (0,1): 2⁰·3¹ = 3        (x₂)
        - (2,0): 2²·3⁰ = 4        (x₁²)
        - (1,1): 2¹·3¹ = 6        (x₁·x₂)
        - (0,2): 2⁰·3² = 9        (x₂²)
        
        Output: [1, 2, 3, 4, 6, 9]
        
        PARAMETERS:
        ----------
        X : Matrix, shape (n_samples, n_features)
            Input features to transform
            
        RETURNS:
        -------
        X_poly : Matrix, shape (n_samples, n_output_features)
            Transformed features with polynomial terms
            
        IMPORTANT:
        ---------
        The number of input features must match what was seen during fit()!
        """
        
        if self.n_input_features_ is None:
            raise RuntimeError("Must call fit() before transform()!")
        
        if X.num_cols != self.n_input_features_:
            raise ValueError(
                f"X has {X.num_cols} features but transformer expects "
                f"{self.n_input_features_} features (from fit)"
            )
        
        # Generate all power combinations
        powers = self._generate_powers()
        
        # Transform each sample
        X_poly_data = []
        for i in range(X.num_rows):
            # Get original features for this sample
            x_original = [X[i, j] for j in range(X.num_cols)]
            
            # Compute polynomial features
            x_poly = []
            for power_tuple in powers:
                # Compute the product: x₁^p₁ · x₂^p₂ · ... · xₙ^pₙ
                feature_value = 1.0
                for feature_idx, power in enumerate(power_tuple):
                    if power > 0:
                        feature_value *= x_original[feature_idx] ** power
                
                x_poly.append(feature_value)
            
            X_poly_data.append(x_poly)
        
        return Matrix(X_poly_data)
    
    def fit_transform(self, X: Matrix) -> Matrix:
        """
        Fit and transform in one step (convenience method).
        
        This is equivalent to:
            transformer.fit(X)
            X_transformed = transformer.transform(X)
            
        But saves you typing!
        
        PARAMETERS:
        ----------
        X : Matrix
            Input features
            
        RETURNS:
        -------
        X_poly : Matrix
            Transformed features
        """
        return self.fit(X).transform(X)
    
    def _generate_powers(self) -> List[Tuple[int, ...]]:
        """
        Generate all combinations of powers up to degree.
        
        This returns a list of tuples, where each tuple represents the powers
        to raise each feature to.
        
        EXAMPLE:
        -------
        2 features, degree=2:
        Returns: [(0,0), (1,0), (0,1), (2,0), (1,1), (0,2)]
        
        Meaning: [1, x₁, x₂, x₁², x₁x₂, x₂²]
        
        THE ALGORITHM:
        -------------
        We use itertools.combinations_with_replacement to generate all ways to
        choose features (with repetition) for each degree level, then flatten.
        
        Actually, a simpler approach: Generate all non-negative integer tuples
        (p₁, p₂, ..., pₙ) such that p₁ + p₂ + ... + pₙ ≤ degree.
        
        RETURNS:
        -------
        powers : List[Tuple[int, ...]]
            List of power tuples, sorted by total degree then lexicographically
        """
        
        n = self.n_input_features_
        powers = []
        
        # Generate all combinations of powers for each degree level
        for total_degree in range(self.degree + 1):
            # Skip degree 0 if we're not including bias
            if total_degree == 0 and not self.include_bias:
                continue
            
            # Generate all ways to distribute total_degree among n features
            # This is equivalent to the "stars and bars" problem
            for combo in self._power_combinations(n, total_degree):
                powers.append(combo)
        
        return powers
    
    def _power_combinations(self, n_features: int, total_degree: int) -> List[Tuple[int, ...]]:
        """
        Generate all ways to distribute total_degree among n_features.
        
        This solves the "stars and bars" combinatorial problem:
        How many ways can you write total_degree as a sum of n_features non-negative integers?
        
        EXAMPLE:
        -------
        n_features=2, total_degree=2:
        Returns: [(0,2), (1,1), (2,0)]
        Meaning: 0x₁+2x₂=2, 1x₁+1x₂=2, 2x₁+0x₂=2
        Or in features: [x₂², x₁x₂, x₁²]
        
        THE ALGORITHM:
        -------------
        We use a recursive approach: To distribute d among n features,
        try giving the first feature 0, 1, 2, ..., d, then recursively
        distribute the remainder among the remaining features.
        
        PARAMETERS:
        ----------
        n_features : int
            Number of features
        total_degree : int
            Sum of powers must equal this
            
        RETURNS:
        -------
        combinations : List[Tuple[int, ...]]
            All valid power tuples
        """
        
        if n_features == 1:
            # Base case: only one feature, must get all the degree
            return [(total_degree,)]
        
        combinations = []
        
        # Try giving first feature 0, 1, 2, ..., total_degree
        for first_power in range(total_degree + 1):
            remaining_degree = total_degree - first_power
            
            # Recursively distribute remaining degree among remaining features
            for rest in self._power_combinations(n_features - 1, remaining_degree):
                combinations.append((first_power,) + rest)
        
        return combinations
    
    def _generate_feature_names(self, powers: List[Tuple[int, ...]]) -> List[str]:
        """
        Generate human-readable names for polynomial features.
        
        This is useful for debugging and understanding what features were created.
        
        EXAMPLE:
        -------
        Powers: [(0,0), (1,0), (0,1), (2,0), (1,1), (0,2)]
        Names: ['1', 'x₀', 'x₁', 'x₀²', 'x₀x₁', 'x₁²']
        
        PARAMETERS:
        ----------
        powers : List[Tuple[int, ...]]
            Power tuples
            
        RETURNS:
        -------
        names : List[str]
            Feature names
        """
        
        names = []
        
        for power_tuple in powers:
            # Special case: all zeros (bias term)
            if all(p == 0 for p in power_tuple):
                names.append('1')
                continue
            
            # Build name from non-zero powers
            terms = []
            for feature_idx, power in enumerate(power_tuple):
                if power == 0:
                    continue
                elif power == 1:
                    terms.append(f'x{feature_idx}')
                else:
                    terms.append(f'x{feature_idx}^{power}')
            
            names.append('·'.join(terms))
        
        return names
    
    def get_feature_names(self) -> List[str]:
        """
        Get names of output features (for inspection).
        
        RETURNS:
        -------
        feature_names : List[str]
            Names of transformed features
        """
        if self.feature_names_ is None:
            raise RuntimeError("Must call fit() first!")
        
        return self.feature_names_