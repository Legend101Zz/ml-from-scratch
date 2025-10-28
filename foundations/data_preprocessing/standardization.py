"""
STANDARDIZATION (Z-SCORE NORMALIZATION): MAKING FEATURES COMPARABLE
===================================================================

Welcome to the most important preprocessing technique in machine learning!

Standardization transforms each feature to have mean=0 and standard deviation=1.
This simple transformation solves one of the biggest problems in ML: features on
wildly different scales that confuse algorithms and prevent learning.

TEACHING PHILOSOPHY:
-------------------
We will understand standardization from three perspectives:
1. STATISTICAL: What it does to data distributions
2. GEOMETRIC: How it affects distances in feature space
3. PRACTICAL: Why algorithms need it and when to apply it

THE PROBLEM IT SOLVES:
----------------------
Imagine training a house price predictor with these features:
- Square footage: ranges from 500 to 5000 (scale of thousands)
- Number of bedrooms: ranges from 1 to 5 (scale of ones)
- Distance to city: ranges from 0 to 50 miles (scale of tens)

Without standardization:
- Gradient descent will be dominated by square footage (largest scale)
- Small changes to bedrooms barely affect the loss
- Learning will zig-zag slowly because gradients have very different magnitudes

With standardization:
- All features contribute equally to distance calculations
- Gradients have similar magnitudes across features
- Convergence is fast and smooth

REAL-WORLD ANALOGY:
------------------
Imagine comparing athletes using different metrics:
- Height in centimeters: 180, 175, 190
- Weight in kilograms: 75, 70, 85
- 100m sprint time in seconds: 11.5, 11.2, 11.8

These are on completely different scales! To compare athletes fairly, we standardize
each metric. Then we can say "this athlete is 1.5 standard deviations above average
in height, 0.5 below average in weight, and 0.8 above average in speed."

THE MATHEMATICAL TRANSFORMATION:
-------------------------------
For each feature j:
1. Compute mean: μ_j = (1/n) Σ x_ij
2. Compute standard deviation: σ_j = sqrt((1/n) Σ (x_ij - μ_j)²)
3. Transform: z_ij = (x_ij - μ_j) / σ_j

After transformation:
- Mean of z_j = 0 (centered at origin)
- Standard deviation of z_j = 1 (unit variance)

WHY THIS WORKS:
--------------
Subtracting the mean centers the data at zero. Dividing by standard deviation
scales it so one unit represents "one standard deviation from the mean."

Now -1.5 means "1.5 standard deviations below average" regardless of the original
units. This makes features directly comparable!

GEOMETRIC INTERPRETATION:
------------------------
In feature space, standardization makes the data cloud more spherical.

Before standardization: elongated ellipsoid (stretched along high-variance features)
After standardization: more spherical (equal spread in all directions)

This spherical shape makes distance metrics meaningful and gradient descent efficient.

WHEN TO USE:
-----------
ALWAYS use standardization for:
- Linear regression (especially with gradient descent)
- Logistic regression
- Neural networks
- Support Vector Machines
- K-Nearest Neighbors
- Any distance-based algorithm

DO NOT standardize for:
- Decision trees and random forests (they're scale-invariant)
- Naive Bayes (works with counts/probabilities directly)

CRITICAL RULES:
--------------
1. FIT on training data ONLY: Learn mean and std from training set
2. TRANSFORM both train and test using training statistics
3. NEVER fit on test data (that's data leakage!)

Example of correct workflow:
```python
scaler = Standardizer()
scaler.fit(X_train)  # Learn mean and std from training data
X_train_scaled = scaler.transform(X_train)  # Apply to training
X_test_scaled = scaler.transform(X_test)   # Apply same transformation to test
```

MACHINE LEARNING IMPACT:
-----------------------
Without standardization:
- Gradient descent: slow convergence, zig-zagging
- Neural networks: unstable training, vanishing/exploding gradients
- KNN: dominated by large-scale features
- Regularization: unfairly penalizes small-scale features

With standardization:
- Gradient descent: fast, smooth convergence
- Neural networks: stable training
- KNN: all features contribute equally
- Regularization: fair penalty across features

Now Let's implement this crucial transformation!
"""

from typing import Optional

from foundations.linear_algebra.vectors_and_matrices import Matrix, Vector
from foundations.statistics.descriptive import *


class Standardizer:
    """
    Standardize features by removing the mean and scaling to unit variance.
    
    The standard score of a sample x is calculated as:
        z = (x - mean) / std
    
    Where mean is the mean of the training samples and std is the standard
    deviation of the training samples.
    
    ATTRIBUTES:
    ----------
    mean_ : Vector or None
        The mean value for each feature in the training set.
        Shape: (n_features)
        
    std_ : Vector or None
        The standard deviation for each feature in the training set.
        Shape: (n_features,)
        
    n_features_ : int or None
        Number of features seen during fit
        
    n_samples_seen_ : int or None
        Number of samples seen during fit (for statistics)
    
    DESIGN NOTES:
    ------------
    We follow scikit-learn's API design:
    1. fit() learns parameters from training data
    2. transform() applies the transformation
    3. fit_transform() does both in one call
    
    This separation is crucial: we learn statistics from training data,
    then apply those same statistics to test data.
    
    WHY WE STORE STATISTICS:
    -----------------------
    In production, you'll receive new data that needs to be transformed
    using the same mean and std you learned during training. You cannot
    recompute mean and std on new data - that would be inconsistent!
    
    Example:
    -------
    >>> X_train = Vector([[1, 2], [3, 4], [5, 6]])
    >>> X_test = Vector([[2, 3], [4, 5]])
    >>> 
    >>> scaler = Standardizer()
    >>> scaler.fit(X_train)
    >>> X_train_scaled = scaler.transform(X_train)
    >>> X_test_scaled = scaler.transform(X_test)  # Uses training mean/std!
    """
    def __init__(self):
        self.mean_ : Optional[Vector] = None
        self.std_ : Optional[Vector] = None
        self.n_features_ : Optional[Vector] = None
    
    def fit(self, X: Matrix) -> 'Standardizer':
        """
        Compute the mean and standard deviation to be used for later scaling.
        
        This method learns the parameters (mean and std) from the training data.
        It does NOT transform the data - that's done by transform().
        
        Parameters:
        ----------
        X : Matrix, shape (n_samples, n_features)
            The training data to compute statistics from.   
        
        Returns:
        -------
        self : Standardizer
            Returns self to allow method chaining:
            scaler.fit(X_train).transform(X_train)
        
        WHAT THIS METHOD DOES:
        ---------------------
        1. Checks input is 2D (rows=samples, columns=features)
        2. Computes mean for each feature (column)
        3. Computes standard deviation for each feature
        4. Stores these statistics for later use in transform()
        
        WHY COMPUTE PER-FEATURE STATISTICS:
        ----------------------------------
        Each feature has its own scale and range. Age might range 0-100,
        income might range 0-1000000. We need separate statistics for each
        to standardize them independently.
        
        Example:
        -------
        >>> X = np.array([[1, 10], [2, 20], [3, 30]])
        >>> scaler = Standardizer()
        >>> scaler.fit(X)
        >>> scaler.mean_
        array([2., 20.])  # Mean of each feature
        >>> scaler.std_
        array([0.816..., 8.165...])  # Std of each feature
        """
        self.n_features_ = X.num_cols
        
        mean_elements = []
        std_elements = []
        
        # Iterate through each feature (column) to compute it's stats
        for j in range(self.n_features_):
            column_vec = X.column(j)
            col_mean = mean(column_vec.elements)
            # Use population standard deviation (ddof=0 in numpy)
            col_std = standard_deviation(column_vec.elements,sample=False)
            
            mean_elements.append(col_mean)
            # Handle constant features (std=0) to avoud division by zero
            std_elements.append(col_std if col_std > 0 else 1.0)
        
        self.mean_ = Vector(mean_elements)
        self.std_ = Vector(std_elements)
        return self
    
    def transform(self, X: Matrix) -> Matrix:
        """
        Perform standardization by centering and scaling.
        
        This method applies the transformation using the mean and std
        learned during fit(). It can be called on training data, validation
        data, or new test data.
        
        Parameters:
        ----------
        X : Matrix, shape (n_samples, n_features)
            The data to transform.
            
        Returns:
        -------
        X_scaled : Matrix
            The transformed data.
        
        THE TRANSFORMATION:
        ------------------
        For each feature j and sample i:
        X_scaled[i, j] = (X[i, j] - mean_j) / std_j
        
        This is applied to all features simultaneously using vectorized operations.
        
        MATHEMATICAL PROPERTIES:
        -----------------------
        After transformation:
        - mean(X_scaled, axis=0) ≈ 0 (numerical precision might give tiny non-zero)
        - std(X_scaled, axis=0) ≈ 1 (ditto)
        
        IMPORTANT: These properties hold exactly for the training data used in fit().
        For new data (test set), mean and std might be slightly different because
        the new data might have a different distribution.
        
        This is CORRECT behavior! We don't want test data to influence our
        transformation. We apply the training transformation consistently.
        
        WHY THIS IS CORRECT FOR TEST DATA:
        ---------------------------------
        Imagine you're predicting house prices. Your training data has houses
        ranging 1000-5000 sqft. You learn mean=3000, std=1000.
        
        Now you get a test house with 6000 sqft. After standardization:
        z = (6000 - 3000) / 1000 = 3.0
        
        This means "3 standard deviations above the training mean."
        This is the correct way to represent this unusually large house!
        
        If you computed new mean/std on test data including this house, you'd
        be leaking information and your test results would be overly optimistic.
        """
        if self.mean_ is None or self.std_ is None:
            raise ValueError("Standardizer has not been fitted yet. Call 'fit' first.")
        if X.num_cols != self.n_features_:
            raise ValueError(f"Input has {X.num_cols} features but scaler was fitted on {self.n_features_}")
        
        scaled_rows = []
        # Since we don't have broadcasting because we wanted to do everything ourselves ahhh... ,So we suffer and iterate and transform element by element     
        
        for i in range(X.num_rows):
            new_row = []
            for j in range(X.num_cols):
                scaled_value = (X[i][j] - self.mean_[j]) / self.std_[j]
                new_row.append(scaled_value)
            scaled_rows.append(new_row)
        
        return Matrix(scaled_rows) 
    
    def fit_transform(self, X:Matrix) -> Matrix:
        """A convenience method to fit to data, then transform it."""
        return self.fit(X).transform(X)
    
    def inverse_transform(self, X_scaled: Matrix) -> Matrix:
        """
        Transform standardized data back to original scale.
        
        This reverses the standardization: X = X_scaled * std + mean
        
        Parameters:
        ----------
        X_scaled : Matrix, shape (n_samples, n_features)
            Standardized data to transform back
        
        Returns:
        -------
        X_original : Matrix, shape (n_samples, n_features)
            Data in original scale
        
        WHY THIS IS USEFUL:
        ------------------
        After training a model on standardized data, predictions are in
        standardized space. To interpret them, you need to convert back
        to original units.
        """
        if self.mean_ is None or self.std_ is None:
            raise ValueError("Standardizer has not been fitted yet.")       
        
        original_rows = []
        for i in range(X_scaled.num_rows):
            new_row = []
            for j in range(X_scaled.num_cols):
                # Reverse the formula: x = z * std + mean
                original_value = (X_scaled[i][j] * self.std_[j]) + self.mean_[j]
                new_row.append(original_value)
            original_rows.append(new_row)
        
        return Matrix(original_rows) 
    
# ==============================================================================
# USAGE EXAMPLES AND TESTS
# ==============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("STANDARDIZATION: INTERACTIVE TUTORIAL (using custom classes)")
    print("=" * 70)
    
    print("\n### THE PROBLEM: FEATURES ON DIFFERENT SCALES ###\n")
    
    # Create example data as Matrix objects
    X_train = Matrix([
        [25.0, 35000.0, 12.0],  # Person 1
        [35.0, 55000.0, 16.0],  # Person 2
        [45.0, 75000.0, 18.0],  # Person 3
        [55.0, 95000.0, 20.0],  # Person 4
    ])
    
    X_test = Matrix([
        [30.0, 45000.0, 14.0],  # New person
        [50.0, 85000.0, 19.0],
    ])
    
    print("Training data (Age, Income, Education):")
    print(X_train)
    
    # Compute distances before standardization using Vector operations
    # Euclidean distance between a and b is the norm of (a - b)
    dist_before = (X_train.row(0) - X_train.row(1)).norm()
    print(f"\nDistance between person 1 and 2 (before): {dist_before:.2f}")
    print("  (Almost entirely due to income difference!)")
    
    print("\n### APPLYING STANDARDIZATION ###\n")
    
    # Create and fit standardizer
    scaler = Standardizer()
    scaler.fit(X_train)
    
    print("Learned from training data:")
    print(f"  Mean: {scaler.mean_}")
    print(f"  Std:  {scaler.std_}")
    
    # Transform training and test data
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("\nStandardized training data:")
    print(X_train_scaled)
    print("\nStandardized test data:")
    print(X_test_scaled)
    
    print("\n### VERIFICATION ###\n")
    # Verify properties of the scaled training data
    scaled_age_col = X_train_scaled.column(0).elements
    scaled_income_col = X_train_scaled.column(1).elements
    scaled_edu_col = X_train_scaled.column(2).elements
    
    print(f"Mean of scaled training data (Age): {mean(scaled_age_col):.2e}")
    print(f"Std of scaled training data (Age): {standard_deviation(scaled_age_col, sample=False):.2f}")
    print("  → Mean should be ~0 and Std should be 1.0 ✓")
    
    # Compute distances after standardization
    dist_after = (X_train_scaled.row(0) - X_train_scaled.row(1)).norm()
    print(f"\nDistance between person 1 and 2 (after): {dist_after:.2f}")
    print("  → Now all features contribute fairly!")
    
    print("\n### INVERSE TRANSFORM ###\n")
    X_reconstructed = scaler.inverse_transform(X_train_scaled)
    print("Reconstructed from standardized:")
    print(X_reconstructed)
    
    # Check if the reconstruction is close to the original
    # We'll check one element to demonstrate
    is_close = abs(X_train[0, 1] - X_reconstructed[0, 1]) < 1e-9
    print(f"\nMatch? (Checking one element): {is_close}")
    
    print("\n✅ Standardization with custom classes complete!")