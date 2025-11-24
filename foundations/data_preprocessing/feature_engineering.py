"""
FEATURE ENGINEERING: The Art of Preparing Data for ML
=====================================================

This module contains utilities for transforming and preparing data before feeding it
to machine learning models. I discovered these weren't optional "nice-to-haves" —
they're absolutely essential for models to work properly!

MY LEARNING JOURNEY:
-------------------
When I first tried training a linear regression model, I just fed it raw data and
wondered why it performed terribly. Then I discovered that ML models are picky eaters!
They need data in specific formats:

1. Numerical features (no strings!)
2. Similar scales (features shouldn't differ by millions in magnitude)
3. Bias terms (column of ones for the intercept)
4. Proper shapes (Matrix objects, not raw lists)

This module implements all those "boring preprocessing steps" that turn out to be
absolutely critical for success.

THE BIG INSIGHT:
---------------
"Garbage in, garbage out" is especially true in machine learning. You can have the
best algorithm in the world, but if your features aren't properly prepared, you'll
get terrible results.

I learned this the hard way: My first linear regression model had a learning rate
that was "too large" — except it wasn't too large, the features just had wildly
different scales! After standardization, the same learning rate worked perfectly.

WHAT WE'LL IMPLEMENT:
--------------------
1. add_bias_column(): Add column of ones for intercept term
2. remove_bias_column(): Remove it when needed
3. standardize(): Scale features to mean=0, std=1
4. normalize(): Scale features to range [0, 1]
5. train_test_split(): Split data for validation (critical!)

WHY THESE MATTER:
----------------
Without these utilities, every model would need to implement its own preprocessing,
leading to code duplication and subtle bugs. Centralizing them here makes everything
cleaner and more reliable.

"""

import sys
import os
import random


from foundations.linear_algebra.vectors_and_matrices import Matrix, Vector
from typing import Tuple, Optional


# ==============================================================================
# BIAS COLUMN OPERATIONS
# ==============================================================================

def add_bias_column(X: Matrix) -> Matrix:
    """
    Add a column of ones to the beginning of the feature matrix.
    
    This is one of the most common operations in machine learning! Almost every
    linear model needs a bias term (intercept), and the standard way to handle it
    is to add a column of ones to X.
    
    WHY DO WE NEED A BIAS COLUMN?
    -----------------------------
    Without bias, our model is: ŷ = w₁x₁ + w₂x₂ + ... + wₙxₙ
    This is a hyperplane that MUST pass through the origin (0, 0, ..., 0).
    
    With bias: ŷ = w₀·1 + w₁x₁ + w₂x₂ + ... + wₙxₙ
    Now the hyperplane can shift anywhere! w₀ controls the vertical shift.
    
    EXAMPLE:
    -------
    Predicting house prices: y = w₀ + w₁·sqft + w₂·bedrooms
    Without w₀, a house with 0 sqft and 0 bedrooms must cost $0!
    With w₀, we can have a base price that all houses start from.
    
    THE TRICK:
    ---------
    By adding a column of ones, we turn the bias into just another weight!
    Now we can write: ŷ = w^T x where x includes the bias feature.
    
    Original X:         With bias:
    [[2, 3],            [[1, 2, 3],
     [4, 5],            [1, 4, 5],
     [6, 7]]            [1, 6, 7]]
    
    The "1" column corresponds to w₀ (the bias weight).
    
    WHEN TO USE:
    -----------
    - Before training almost any linear model
    - Linear regression, logistic regression, perceptron, etc.
    - NOT needed if your model handles bias separately
    
    PARAMETERS:
    ----------
    X : Matrix, shape (n_samples, n_features)
        Original feature matrix without bias
        
    RETURNS:
    -------
    X_with_bias : Matrix, shape (n_samples, n_features + 1)
        Feature matrix with column of ones prepended
        
    EXAMPLE:
    -------
    >>> X = Matrix([[2, 3], [4, 5]])
    >>> X_bias = add_bias_column(X)
    >>> print(X_bias.elements)
    [[1, 2, 3], [1, 4, 5]]
    """
    
    n_samples = X.num_rows
    n_features = X.num_cols
    
    # Create new matrix with one extra column
    new_data = []
    for i in range(n_samples):
        # Start with 1 (bias), then add original features
        row = [1.0]  # The bias feature is always 1
        for j in range(n_features):
            row.append(X[i, j])
        new_data.append(row)
    
    return Matrix(new_data)


def remove_bias_column(X: Matrix) -> Matrix:
    """
    Remove the first column from a feature matrix.
    
    This is useful when you've added a bias column but need to remove it later,
    for example when visualizing or when switching to a model that handles bias
    differently.
    
    WHEN TO USE:
    -----------
    - When converting between representations
    - Before certain visualizations
    - When you accidentally added bias twice (oops!)
    
    PARAMETERS:
    ----------
    X : Matrix, shape (n_samples, n_features)
        Feature matrix with bias column as first column
        
    RETURNS:
    -------
    X_no_bias : Matrix, shape (n_samples, n_features - 1)
        Feature matrix without first column
        
    EXAMPLE:
    -------
    >>> X_bias = Matrix([[1, 2, 3], [1, 4, 5]])
    >>> X = remove_bias_column(X_bias)
    >>> print(X.elements)
    [[2, 3], [4, 5]]
    """
    
    if X.num_cols < 2:
        raise ValueError("Cannot remove bias column: matrix only has one column!")
    
    n_samples = X.num_rows
    n_features = X.num_cols
    
    # Create new matrix without first column
    new_data = []
    for i in range(n_samples):
        row = [X[i, j] for j in range(1, n_features)]  # Skip column 0
        new_data.append(row)
    
    return Matrix(new_data)


def has_bias_column(X: Matrix, tolerance: float = 1e-6) -> bool:
    """
    Check if the first column of X is all ones (bias column).
    
    This is useful for detecting whether bias has already been added, preventing
    us from accidentally adding it twice!
    
    THE CHECK:
    ---------
    We check if every element in the first column is within tolerance of 1.0.
    We use tolerance instead of exact equality to handle floating point errors.
    
    PARAMETERS:
    ----------
    X : Matrix
        Feature matrix to check
    tolerance : float
        How close to 1.0 counts as "one" (handles floating point errors)
        
    RETURNS:
    -------
    has_bias : bool
        True if first column appears to be all ones
        
    EXAMPLE:
    -------
    >>> X_with = Matrix([[1, 2, 3], [1, 4, 5]])
    >>> has_bias_column(X_with)
    True
    
    >>> X_without = Matrix([[2, 3], [4, 5]])
    >>> has_bias_column(X_without)
    False
    """
    
    if X.num_cols == 0:
        return False
    
    # Check if all elements in first column are close to 1
    for i in range(X.num_rows):
        if abs(X[i, 0] - 1.0) > tolerance:
            return False
    
    return True


# ==============================================================================
# FEATURE SCALING
# ==============================================================================

class Standardizer:
    """
    Standardize features to have mean=0 and standard deviation=1.
    
    This is one of the most important preprocessing steps! When features have
    wildly different scales, gradient descent struggles.
    
    WHY STANDARDIZATION MATTERS:
    ---------------------------
    Imagine predicting house prices from [square_feet, num_bedrooms]:
    - square_feet: ranges from 1000 to 5000 (scale: thousands)
    - num_bedrooms: ranges from 1 to 5 (scale: ones)
    
    Without standardization:
    - The loss surface is elongated (stretched in sqft dimension)
    - Gradient descent zigzags, converges slowly
    - Need different learning rates for different features
    
    After standardization (both features have mean≈0, std≈1):
    - Loss surface is more circular (balanced)
    - Gradient descent converges quickly, directly toward minimum
    - Same learning rate works for all features
    
    THE FORMULA:
    -----------
    z = (x - mean) / std
    
    This transforms any distribution to have mean=0 and std=1.
    
    IMPORTANT: FIT ON TRAINING DATA ONLY!
    -------------------------------------
    We compute mean and std from the TRAINING set, then apply those same values
    to the test set. This prevents data leakage!
    
    Wrong approach:
```
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    scaler.fit(X_test)  # WRONG! This is data leakage!
    X_test_scaled = scaler.transform(X_test)
```
    
    Right approach:
```
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)  # Use training statistics!
```
    
    ATTRIBUTES:
    ----------
    mean_ : Vector
        Mean of each feature (computed during fit)
        
    std_ : Vector
        Standard deviation of each feature (computed during fit)
        
    EXAMPLE USAGE:
    -------------
    >>> X_train = Matrix([[1000, 3], [2000, 4], [1500, 2]])
    >>> 
    >>> scaler = Standardizer()
    >>> scaler.fit(X_train)
    >>> X_train_scaled = scaler.transform(X_train)
    >>> # Now all features have mean≈0, std≈1
    """
    
    def __init__(self):
        self.mean_ = None
        self.std_ = None
    
    def fit(self, X: Matrix) -> 'Standardizer':
        """
        Compute mean and standard deviation for each feature.
        
        PARAMETERS:
        ----------
        X : Matrix, shape (n_samples, n_features)
            Training data to compute statistics from
            
        RETURNS:
        -------
        self : Standardizer
            Returns self for method chaining
        """
        
        n_samples = X.num_rows
        n_features = X.num_cols
        
        # Compute mean for each feature
        means = []
        for j in range(n_features):
            col = X.column(j)
            col_mean = sum(col.elements) / n_samples
            means.append(col_mean)
        
        self.mean_ = Vector(means)
        
        # Compute standard deviation for each feature
        stds = []
        for j in range(n_features):
            col = X.column(j)
            # Variance = average of squared differences from mean
            squared_diffs = [(val - self.mean_[j]) ** 2 for val in col.elements]
            variance = sum(squared_diffs) / n_samples
            std = variance ** 0.5
            
            # Handle zero variance (constant feature)
            if std < 1e-8:
                std = 1.0  # Don't divide by zero; keep feature as-is
            
            stds.append(std)
        
        self.std_ = Vector(stds)
        
        return self
    
    def transform(self, X: Matrix) -> Matrix:
        """
        Standardize features using previously computed mean and std.
        
        PARAMETERS:
        ----------
        X : Matrix
            Data to transform
            
        RETURNS:
        -------
        X_scaled : Matrix
            Standardized data with mean≈0, std≈1
        """
        
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Must call fit() before transform()!")
        
        if X.num_cols != len(self.mean_):
            raise ValueError(
                f"X has {X.num_cols} features but scaler was fitted with "
                f"{len(self.mean_)} features"
            )
        
        # Apply: z = (x - mean) / std
        scaled_data = []
        for i in range(X.num_rows):
            row = []
            for j in range(X.num_cols):
                scaled_val = (X[i, j] - self.mean_[j]) / self.std_[j]
                row.append(scaled_val)
            scaled_data.append(row)
        
        return Matrix(scaled_data)
    
    def fit_transform(self, X: Matrix) -> Matrix:
        """
        Fit and transform in one step (convenience method).
        
        Equivalent to: scaler.fit(X).transform(X)
        
        PARAMETERS:
        ----------
        X : Matrix
            Training data
            
        RETURNS:
        -------
        X_scaled : Matrix
            Standardized training data
        """
        return self.fit(X).transform(X)
    
    def inverse_transform(self, X_scaled: Matrix) -> Matrix:
        """
        Reverse the standardization: x = z * std + mean
        
        This is useful for interpreting results in the original scale.
        
        PARAMETERS:
        ----------
        X_scaled : Matrix
            Standardized data
            
        RETURNS:
        -------
        X_original : Matrix
            Data in original scale
        """
        
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Must call fit() first!")
        
        # Apply: x = z * std + mean
        original_data = []
        for i in range(X_scaled.num_rows):
            row = []
            for j in range(X_scaled.num_cols):
                original_val = X_scaled[i, j] * self.std_[j] + self.mean_[j]
                row.append(original_val)
            original_data.append(row)
        
        return Matrix(original_data)


class MinMaxScaler:
    """
    Scale features to a fixed range [min_val, max_val], typically [0, 1].
    
    This is an alternative to standardization. Instead of making mean=0 and std=1,
    we scale features to a specific range.
    
    THE FORMULA:
    -----------
    x_scaled = (x - x_min) / (x_max - x_min)  # Scales to [0, 1]
    
    Then: x_scaled = x_scaled * (max_val - min_val) + min_val  # Scale to [min_val, max_val]
    
    WHEN TO USE MIN-MAX VS STANDARDIZATION:
    --------------------------------------
    Use Min-Max when:
    - You need features in a specific range (e.g., [0, 1] for neural networks)
    - Data has known bounds (e.g., age, percentages)
    - You want to preserve zero values
    
    Use Standardization when:
    - Data has no clear bounds
    - Data has outliers (min-max is sensitive to outliers!)
    - You're using algorithms sensitive to feature scale (gradient descent)
    
    THE OUTLIER PROBLEM:
    -------------------
    If your data is [1, 2, 3, 4, 100], min-max scaling gives:
    [0, 0.01, 0.02, 0.03, 1]
    
    The outlier (100) compressed everything else to near-zero!
    Standardization handles this better.
    
    PARAMETERS:
    ----------
    feature_range : Tuple[float, float]
        Desired range (min, max) for scaled features
        Default: (0, 1)
        
    ATTRIBUTES:
    ----------
    data_min_ : Vector
        Minimum value of each feature (from training data)
        
    data_max_ : Vector
        Maximum value of each feature (from training data)
    """
    
    def __init__(self, feature_range: Tuple[float, float] = (0, 1)):
        self.feature_range = feature_range
        self.data_min_ = None
        self.data_max_ = None
    
    def fit(self, X: Matrix) -> 'MinMaxScaler':
        """
        Compute min and max for each feature.
        
        PARAMETERS:
        ----------
        X : Matrix
            Training data
            
        RETURNS:
        -------
        self : MinMaxScaler
        """
        
        n_features = X.num_cols
        
        mins = []
        maxs = []
        
        for j in range(n_features):
            col = X.column(j)
            col_min = min(col.elements)
            col_max = max(col.elements)
            
            # Handle constant feature
            if abs(col_max - col_min) < 1e-8:
                col_max = col_min + 1.0
            
            mins.append(col_min)
            maxs.append(col_max)
        
        self.data_min_ = Vector(mins)
        self.data_max_ = Vector(maxs)
        
        return self
    
    def transform(self, X: Matrix) -> Matrix:
        """
        Scale features to [min_val, max_val].
        
        PARAMETERS:
        ----------
        X : Matrix
            Data to scale
            
        RETURNS:
        -------
        X_scaled : Matrix
            Scaled data
        """
        
        if self.data_min_ is None or self.data_max_ is None:
            raise RuntimeError("Must call fit() before transform()!")
        
        min_val, max_val = self.feature_range
        scale = max_val - min_val
        
        scaled_data = []
        for i in range(X.num_rows):
            row = []
            for j in range(X.num_cols):
                # First scale to [0, 1]
                normalized = (X[i, j] - self.data_min_[j]) / (self.data_max_[j] - self.data_min_[j])
                # Then scale to [min_val, max_val]
                scaled_val = normalized * scale + min_val
                row.append(scaled_val)
            scaled_data.append(row)
        
        return Matrix(scaled_data)
    
    def fit_transform(self, X: Matrix) -> Matrix:
        """Fit and transform in one step."""
        return self.fit(X).transform(X)
    
    def inverse_transform(self, X_scaled: Matrix) -> Matrix:
        """Reverse the scaling to original values."""
        
        if self.data_min_ is None or self.data_max_ is None:
            raise RuntimeError("Must call fit() first!")
        
        min_val, max_val = self.feature_range
        scale = max_val - min_val
        
        original_data = []
        for i in range(X_scaled.num_rows):
            row = []
            for j in range(X_scaled.num_cols):
                # Reverse: first to [0, 1]
                normalized = (X_scaled[i, j] - min_val) / scale
                # Then to original scale
                original_val = normalized * (self.data_max_[j] - self.data_min_[j]) + self.data_min_[j]
                row.append(original_val)
            original_data.append(row)
        
        return Matrix(original_data)


# ==============================================================================
# TRAIN-TEST SPLIT
# ==============================================================================

def train_test_split(
    X: Matrix,
    y: Matrix,
    test_size: float = 0.2,
    random_state: Optional[int] = None,
    shuffle: bool = True
) -> Tuple[Matrix, Matrix, Matrix, Matrix]:
    """
    Split data into training and test sets.
    
    This is ESSENTIAL for validating your model! If you train and test on the same
    data, you have no idea if your model generalizes to new data.
    
    WHY WE NEED THIS:
    ----------------
    The goal of machine learning is to make predictions on NEW, unseen data.
    If we only evaluate on training data, we might think our model is great when
    it actually just memorized the training examples (overfitting).
    
    The solution: Hold out some data for testing!
    1. Train on 80% of data (training set)
    2. Evaluate on remaining 20% (test set)
    3. Test set simulates "new, unseen data"
    
    THE GOLDEN RULE:
    ---------------
    NEVER use test data during training! Not for:
    - Computing features statistics (mean, std)
    - Tuning hyperparameters
    - Selecting features
    - Making any decisions
    
    Test data is sacred — only touch it once at the very end to get final performance.
    
    THE SHUFFLE PARAMETER:
    ---------------------
    shuffle=True (default): Randomly shuffle before splitting
    shuffle=False: Split in order (first 80% train, last 20% test)
    
    When to use shuffle=False:
    - Time series data (can't shuffle or you break temporal ordering!)
    - Data is already randomly ordered
    - You specifically want first N samples for training
    
    Otherwise, always shuffle! If data is ordered (e.g., all class 0 then all class 1),
    non-shuffled split would be disastrous.
    
    PARAMETERS:
    ----------
    X : Matrix, shape (n_samples, n_features)
        Feature matrix
        
    y : Matrix, shape (n_samples, 1)
        Target vector
        
    test_size : float, default=0.2
        Fraction of data to use for testing
        Common values: 0.2 (20%), 0.25 (25%), 0.3 (30%)
        
    random_state : int or None
        Random seed for reproducibility
        If None, split is different each time
        If set (e.g., 42), split is consistent
        
    shuffle : bool, default=True
        Whether to shuffle data before splitting
        
    RETURNS:
    -------
    X_train : Matrix
        Training features
        
    X_test : Matrix
        Test features
        
    y_train : Matrix
        Training targets
        
    y_test : Matrix
        Test targets
        
    EXAMPLE:
    -------
    >>> X = Matrix([[1,2], [3,4], [5,6], [7,8], [9,10]])
    >>> y = Matrix([[1], [2], [3], [4], [5]])
    >>> 
    >>> X_train, X_test, y_train, y_test = train_test_split(
    ...     X, y, test_size=0.2, random_state=42
    ... )
    >>> 
    >>> print(f"Training samples: {X_train.num_rows}")
    >>> print(f"Test samples: {X_test.num_rows}")
    """
    
    if X.num_rows != y.num_rows:
        raise ValueError(
            f"X has {X.num_rows} samples but y has {y.num_rows} samples"
        )
    
    if not 0 < test_size < 1:
        raise ValueError(f"test_size must be between 0 and 1, got {test_size}")
    
    n_samples = X.num_rows
    n_test = int(n_samples * test_size)
    n_train = n_samples - n_test
    
    if n_test == 0:
        raise ValueError(
            f"test_size={test_size} results in 0 test samples for {n_samples} samples"
        )
    
    # Create indices
    indices = list(range(n_samples))
    
    # Shuffle if requested
    if shuffle:
        if random_state is not None:
            random.seed(random_state)
        random.shuffle(indices)
    
    # Split indices
    train_indices = indices[:n_train]
    test_indices = indices[n_train:]
    
    # Create train and test sets
    X_train_data = [X.elements[i] for i in train_indices]
    X_test_data = [X.elements[i] for i in test_indices]
    y_train_data = [y.elements[i] for i in train_indices]
    y_test_data = [y.elements[i] for i in test_indices]
    
    X_train = Matrix(X_train_data)
    X_test = Matrix(X_test_data)
    y_train = Matrix(y_train_data)
    y_test = Matrix(y_test_data)
    
    return X_train, X_test, y_train, y_test


# ==============================================================================
# EXAMPLE USAGE
# ==============================================================================

if __name__ == "__main__":
    """
    Demonstrate all feature engineering utilities.
    """
    
    print("""
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║      FEATURE ENGINEERING: PREPARING DATA FOR ML SUCCESS         ║
║                                                                  ║
║  "I spent 2 weeks implementing algorithms and 10 minutes on    ║
║   preprocessing. Then I spent 2 days fixing preprocessing!"     ║
║   - Me, learning the hard way                                   ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
    """)
    
    # =========================================================================
    # Demo 1: Bias Column
    # =========================================================================
    
    print("\n" + "=" * 70)
    print("DEMO 1: ADDING BIAS COLUMN")
    print("=" * 70)
    
    X_no_bias = Matrix([[2, 3], [4, 5], [6, 7]])
    print("\nOriginal features (no bias):")
    for i in range(X_no_bias.num_rows):
        print(f"  {X_no_bias.elements[i]}")
    
    X_with_bias = add_bias_column(X_no_bias)
    print("\nWith bias column added:")
    for i in range(X_with_bias.num_rows):
        print(f"  {X_with_bias.elements[i]}")
    
    print(f"\nShape changed: {X_no_bias.shape} → {X_with_bias.shape}")
    print("The first column (all ones) represents the bias feature!")
    
    # Check detection
    print(f"\nhas_bias_column(X_with_bias): {has_bias_column(X_with_bias)}")
    print(f"has_bias_column(X_no_bias): {has_bias_column(X_no_bias)}")
    
    # =========================================================================
    # Demo 2: Standardization
    # =========================================================================
    
    print("\n" + "=" * 70)
    print("DEMO 2: FEATURE STANDARDIZATION")
    print("=" * 70)
    
    # Create data with very different scales
    X_unscaled = Matrix([
        [1000, 3],  # square feet, bedrooms
        [2000, 4],
        [1500, 2],
        [2500, 5],
        [1800, 3]
    ])
    
    print("\nOriginal data (different scales):")
    print("  sqft  | bedrooms")
    print("--------|----------")
    for i in range(X_unscaled.num_rows):
        print(f"  {X_unscaled[i,0]:5.0f} |    {X_unscaled[i,1]:.0f}")
    
    # Compute statistics manually for demonstration
    sqft_values = [X_unscaled[i,0] for i in range(X_unscaled.num_rows)]
    sqft_mean = sum(sqft_values) / len(sqft_values)
    sqft_std = (sum((x - sqft_mean)**2 for x in sqft_values) / len(sqft_values)) ** 0.5
    
    bed_values = [X_unscaled[i,1] for i in range(X_unscaled.num_rows)]
    bed_mean = sum(bed_values) / len(bed_values)
    bed_std = (sum((x - bed_mean)**2 for x in bed_values) / len(bed_values)) ** 0.5
    
    print(f"\nBefore standardization:")
    print(f"  sqft: mean={sqft_mean:.1f}, std={sqft_std:.1f}")
    print(f"  beds: mean={bed_mean:.1f}, std={bed_std:.1f}")
    print("  → Very different scales! Gradient descent will struggle.")
    
    # Standardize
    scaler = Standardizer()
    X_scaled = scaler.fit_transform(X_unscaled)
    
    print("\nAfter standardization:")
    print("  sqft  | bedrooms")
    print("--------|----------")
    for i in range(X_scaled.num_rows):
        print(f"  {X_scaled[i,0]:5.2f} |  {X_scaled[i,1]:5.2f}")
    
    # Verify standardization worked
    scaled_sqft = [X_scaled[i,0] for i in range(X_scaled.num_rows)]
    scaled_mean = sum(scaled_sqft) / len(scaled_sqft)
    scaled_std = (sum(x**2 for x in scaled_sqft) / len(scaled_sqft)) ** 0.5
    
    print(f"\nAfter standardization:")
    print(f"  mean ≈ {scaled_mean:.6f} (should be ≈0)")
    print(f"  std ≈ {scaled_std:.6f} (should be ≈1)")
    print("  ✓ All features now have similar scales!")
    
    # =========================================================================
    # Demo 3: Min-Max Scaling
    # =========================================================================
    
    print("\n" + "=" * 70)
    print("DEMO 3: MIN-MAX SCALING TO [0, 1]")
    print("=" * 70)
    
    X_data = Matrix([[100], [200], [150], [250], [175]])
    
    print("\nOriginal data:")
    for i in range(X_data.num_rows):
        print(f"  {X_data[i,0]:.0f}")
    
    minmax_scaler = MinMaxScaler(feature_range=(0, 1))
    X_minmax = minmax_scaler.fit_transform(X_data)
    
    print("\nAfter min-max scaling to [0, 1]:")
    for i in range(X_minmax.num_rows):
        original = X_data[i,0]
        scaled = X_minmax[i,0]
        print(f"  {original:.0f} → {scaled:.3f}")
    
    print("\n✓ Smallest value → 0, largest value → 1")
    
    # =========================================================================
    # Demo 4: Train-Test Split
    # =========================================================================
    
    print("\n" + "=" * 70)
    print("DEMO 4: TRAIN-TEST SPLIT")
    print("=" * 70)
    
    # Create dataset
    X_full = Matrix([[i, i*2] for i in range(1, 11)])
    y_full = Matrix([[i*3] for i in range(1, 11)])
    
    print(f"\nFull dataset: {X_full.num_rows} samples")
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_full, y_full,
        test_size=0.3,  # 30% for testing
        random_state=42,
        shuffle=True
    )
    
    print(f"\nAfter split (test_size=0.3):")
    print(f"  Training set: {X_train.num_rows} samples ({X_train.num_rows/X_full.num_rows*100:.0f}%)")
    print(f"  Test set: {X_test.num_rows} samples ({X_test.num_rows/X_full.num_rows*100:.0f}%)")
    
    print("\nTrain on training set, evaluate on test set!")
    print("Test set simulates 'new, unseen data'")
    
    # =========================================================================
    # Demo 5: Complete Preprocessing Pipeline
    # =========================================================================
    
    print("\n" + "=" * 70)
    print("DEMO 5: COMPLETE PREPROCESSING PIPELINE")
    print("=" * 70)
    
    # Raw data
    X_raw = Matrix([
        [1000, 3],
        [2000, 4],
        [1500, 2],
        [2500, 5]
    ])
    y_raw = Matrix([[300], [450], [350], [550]])
    
    print("\n1. Original data (raw features):")
    print(f"   Shape: {X_raw.shape}")
    
    # Step 1: Train-test split (BEFORE any preprocessing!)
    X_train, X_test, y_train, y_test = train_test_split(
        X_raw, y_raw, test_size=0.25, random_state=42
    )
    
    print(f"\n2. Split into train/test:")
    print(f"   Train: {X_train.num_rows} samples")
    print(f"   Test: {X_test.num_rows} samples")
    
    # Step 2: Standardize (fit on train only!)
    scaler2 = Standardizer()
    X_train_scaled = scaler2.fit_transform(X_train)
    X_test_scaled = scaler2.transform(X_test)  # Use training statistics!
    
    print(f"\n3. Standardized features:")
    print(f"   Fitted on training data only")
    print(f"   Applied same transformation to test data")
    
    # Step 3: Add bias
    X_train_final = add_bias_column(X_train_scaled)
    X_test_final = add_bias_column(X_test_scaled)
    
    print(f"\n4. Added bias column:")
    print(f"   Final train shape: {X_train_final.shape}")
    print(f"   Final test shape: {X_test_final.shape}")
    
    print("\n✓ Data is now ready for model training!")
    print("\nThe pipeline: raw → split → scale → add bias → train")
    
    # =========================================================================
    # Key Takeaways
    # =========================================================================
    
    print("\n" + "=" * 70)
    print("KEY TAKEAWAYS")
    print("=" * 70)
    print("""
1. BIAS COLUMN: Always add it for linear models
   - Allows intercept term (vertical shift)
   - Add before training, remove for visualization

2. STANDARDIZATION: Essential for gradient descent
   - Transforms to mean=0, std=1
   - Makes all features same scale
   - Speeds up convergence dramatically

3. MIN-MAX SCALING: Alternative to standardization
   - Scales to specific range (usually [0,1])
   - Good for bounded features
   - Sensitive to outliers

4. TRAIN-TEST SPLIT: Critical for validation!
   - Never evaluate on training data
   - Common splits: 80/20, 70/30, 75/25
   - Always split BEFORE preprocessing

5. THE GOLDEN RULE: Fit on training data only!
```
   scaler.fit(X_train)           # Learn from training
   X_train = scaler.transform(X_train)  # Apply to training
   X_test = scaler.transform(X_test)    # Apply to test (using training stats!)
```

6. PREPROCESSING ORDER:
   a. Split data (train/test)
   b. Fit scaler on training data
   c. Transform both train and test
   d. Add bias column
   e. Train model

Proper preprocessing isn't optional — it's the difference between a model
that works and one that fails mysteriously!
    """)